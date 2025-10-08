import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from dataset_classes.py import HbImageDataset, UnlabelledImageDataset, CombinedImageDataset
from utils.py import make_run_dir
from torchvision import transforms, models

# ---------------------------
# Transforms
# ---------------------------
ssl_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # transforms.ColorJitter(0.4,0.4,0.4,0.1), => May be too strong for hb predictions
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def run_ssl_pipeline_kfold_with_r2(labeled_df,
                                   unlabelled_dir=None,
                                   ssl_epochs=20,
                                   ssl_batch=8,
                                   fine_tune_backbone=True,
                                   fine_tune_epochs=10,
                                   use_metadata=False,
                                   optuna_trials=20,
                                   run_base_dir="models",
                                   load_run_dir=None,
                                   n_splits=5):

    run_dir = load_run_dir if load_run_dir else make_run_dir(run_base_dir)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # -------------------
    # Load or pretrain SSL backbone
    # -------------------
    backbone = get_ssl_backbone(pretrained=False)
    proj_head = ProjectionHead().to(device)

    if load_run_dir and os.path.exists(os.path.join(run_dir, "ssl_backbone_state_dict.pth")):
        backbone.load_state_dict(torch.load(os.path.join(run_dir, "ssl_backbone_state_dict.pth"), map_location=device))
        proj_head.load_state_dict(torch.load(os.path.join(run_dir, "ssl_projection_head_state_dict.pth"), map_location=device))
        backbone.to(device).eval()
        proj_head.to(device).eval()
        print("Loaded pretrained backbone and projection head from saved run.")
    else:
        print("=== SSL Pretraining ===")
        if unlabelled_dir:
            backbone, proj_head, ssl_losses = pretrain_ssl(unlabelled_dir, ssl_transform,
                                                           epochs=ssl_epochs, batch_size=ssl_batch,
                                                           run_dir=run_dir, unlabelled=True)
        else:
            backbone, proj_head, ssl_losses = pretrain_ssl(labeled_df, ssl_transform,
                                                           epochs=ssl_epochs, batch_size=ssl_batch,
                                                           run_dir=run_dir, unlabelled=False)

    # -------------------
    # K-Fold CV
    # -------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(labeled_df)):
        print(f"\n--- K-Fold {fold_idx+1}/{n_splits} ---")
        train_df, test_df = labeled_df.iloc[train_idx], labeled_df.iloc[test_idx]

        # Fine-tune backbone if required
        if fine_tune_backbone:
            dataset = HbImageDataset(train_df, transform=ssl_transform, n_views=2)
            loader = DataLoader(dataset, batch_size=ssl_batch, shuffle=True, num_workers=2, pin_memory=True)
            backbone.train()
            proj_head.train()
            optimizer = torch.optim.Adam(list(backbone.parameters()) + list(proj_head.parameters()), lr=1e-4)
            for epoch in range(fine_tune_epochs):
                total_loss = 0.0
                for views, _ in loader:
                    v1, v2 = views[:,0].to(device), views[:,1].to(device)
                    feats1, feats2 = backbone(v1).view(v1.size(0), -1), backbone(v2).view(v2.size(0), -1)
                    z1, z2 = proj_head(feats1), proj_head(feats2)
                    loss = nt_xent_loss(z1, z2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Fold {fold_idx+1} Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

        # Extract embeddings
        X_train, y_train = extract_embeddings(train_df, backbone, val_transform)
        X_test, y_test = extract_embeddings(test_df, backbone, val_transform)

        # Optuna hyperparameter tuning
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators",50,300),
                "max_depth": trial.suggest_int("max_depth",2,10),
                "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
                "subsample": trial.suggest_float("subsample",0.5,1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1.0),
                "random_state": RANDOM_SEED,
                "verbosity": 0,
                "n_jobs": 1,
                "tree_method": "hist"
            }
            kf_inner = KFold(n_splits=min(5, len(train_df)), shuffle=True, random_state=RANDOM_SEED)
            maes = []
            for tr_idx, val_idx in kf_inner.split(X_train):
                model = xgb.XGBRegressor(**params)
                model.fit(X_train[tr_idx], y_train[tr_idx])
                y_pred = model.predict(X_train[val_idx])
                maes.append(mean_absolute_error(y_train[val_idx], y_pred))
            return np.mean(maes)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best_params = study.best_params
        print(f"Best params fold {fold_idx+1}: {best_params}")

        # Train final model
        final_model = xgb.XGBRegressor(**best_params, random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist")
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        # Compute metrics including R²
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"[Fold {fold_idx+1}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        fold_metrics.append({"fold": fold_idx+1, "RMSE": rmse, "MAE": mae, "R2": r2})

    # -------------------
    # Summary metrics
    # -------------------
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.loc["Mean"] = metrics_df.mean()
    print("\n=== Fold Metrics Summary ===")
    print(metrics_df)

    # Save metrics and models
    metrics_df.to_csv(os.path.join(run_dir, "kfold_metrics.csv"), index=False)
    torch.save(backbone.state_dict(), os.path.join(run_dir, "ssl_backbone_state_dict.pth"))
    torch.save(proj_head.state_dict(), os.path.join(run_dir, "ssl_projection_head_state_dict.pth"))

    # Visualization
    plt.figure(figsize=(10,6))
    plt.plot(metrics_df['fold'][:-1], metrics_df['RMSE'][:-1], marker='o', label='RMSE')
    plt.plot(metrics_df['fold'][:-1], metrics_df['MAE'][:-1], marker='s', label='MAE')
    plt.plot(metrics_df['fold'][:-1], metrics_df['R2'][:-1], marker='^', label='R²')
    plt.axhline(metrics_df.loc["Mean", 'RMSE'], color='blue', linestyle='--', alpha=0.5)
    plt.axhline(metrics_df.loc["Mean", 'MAE'], color='orange', linestyle='--', alpha=0.5)
    plt.axhline(metrics_df.loc["Mean", 'R2'], color='green', linestyle='--', alpha=0.5)
    plt.xlabel("Fold")
    plt.ylabel("Metric Value")
    plt.title("K-Fold Regression Metrics (RMSE, MAE, R²)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "kfold_metrics_plot.png"))
    plt.show()

    return {
        "backbone": backbone,
        "proj_head": proj_head,
        "metrics_df": metrics_df,
        "run_dir": run_dir
    }
