import pandas as pd
import os
 
def compute_breakage_rate(df):
    df = df.copy()
    df["total_eggs_loaded"] = df["total_eggs_loaded"].fillna(0).astype(float)
    df["total_eggs_broken"] = df["total_eggs_broken"].fillna(0).astype(float)
    # safe division
    df["breakage_rate"] = df.apply(
        lambda r: (r["total_eggs_broken"]/r["total_eggs_loaded"]) if r["total_eggs_loaded"]>0 else 0,
        axis=1
    )
    # month feature for trend
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df
 
def save_processed(df, name, outdir="data/processed"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path