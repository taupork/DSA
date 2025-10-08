from datatime import datetime
import pandas as pd
import json 
import os

# ---------------------------
# Utilities
# ---------------------------
def make_run_dir(base="models", prefix="run"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def safe_save_json(df, path):
    try:
        df.to_json(path, orient="records", lines=True)
    except Exception:
        df.to_csv(path.replace(".json", ".csv"), index=False)