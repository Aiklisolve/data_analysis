import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.stats import t
from dotenv import load_dotenv

load_dotenv()
egg_price_inr = float(os.getenv("PRICE_PER_EGG", 5.0))  # fallback if not set

# === ANOVA Filtering ===
def get_significant_anova_factors(anova_csv_path: str, p_threshold: float = 0.05) -> list:
    if not os.path.exists(anova_csv_path):
        return []
    df = pd.read_csv(anova_csv_path)
    # Handle index column if saved as "Unnamed: 0"
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "factor"})
    elif df.columns[0] != "factor":
        df = df.rename(columns={df.columns[0]: "factor"})
    significant = df[df["PR(>F)"] < p_threshold]["factor"].tolist()
    return [f.split("(")[-1].split(")")[0] if "(" in f else f for f in significant]

# === RCAS Computation ===
def assign_confidence(n):
    if n >= 30: return 'HIGH'
    elif n >= 20: return 'MEDIUM'
    else: return 'LOW'

def compute_rcas(df: pd.DataFrame, group_col: str, label_map: dict = None, mode: str = "raw", egg_price: float = None) -> pd.DataFrame:
    if mode == "raw":
        grouped = df.groupby(group_col).agg(
            deliveries_count=('delivery_id', 'count'),
            eggs_loaded=('total_eggs_loaded', 'sum'),
            eggs_broken=('total_eggs_broken', 'sum')
        ).reset_index()
    else:
        grouped = df.copy()
        grouped['eggs_loaded'] = grouped['total_eggs_loaded']
        grouped['eggs_broken'] = grouped['total_eggs_broken']

    grouped['breakage_rate'] = grouped['eggs_broken'] / grouped['eggs_loaded']

    fleet_avg = grouped['eggs_broken'].sum() / grouped['eggs_loaded'].sum()
    fleet_std = grouped['breakage_rate'].std(ddof=0)

    grouped['z_score'] = (grouped['breakage_rate'] - fleet_avg) / fleet_std
    grouped['weight'] = np.log1p(grouped['eggs_loaded'])
    grouped['rcas'] = grouped['z_score'] * grouped['weight']

    rcas_min = grouped['rcas'].min()
    rcas_max = grouped['rcas'].max()
    grouped['rcas_normalized'] = (grouped['rcas'] - rcas_min) / (rcas_max - rcas_min)
    grouped['rcas_rank'] = grouped['rcas_normalized'].rank(method='dense', ascending=False).astype(int)

    grouped['effect_size'] = grouped['breakage_rate'] - fleet_avg
    grouped['confidence_level'] = grouped['deliveries_count'].apply(assign_confidence)

    grouped['ci_margin'] = grouped.apply(
        lambda row: t.ppf(0.975, row['deliveries_count'] - 1) *
                    (fleet_std / np.sqrt(row['deliveries_count']))
                    if row['deliveries_count'] > 1 else 0,
        axis=1
    )
    grouped['ci_lower'] = grouped['breakage_rate'] - grouped['ci_margin']
    grouped['ci_upper'] = grouped['breakage_rate'] + grouped['ci_margin']

    grouped['percentile_rank'] = grouped['breakage_rate'].rank(pct=True) * 100
    grouped['percentile_rank'] = grouped['percentile_rank'].astype(int)

    if egg_price:
        grouped['cost_impact_inr'] = grouped['eggs_broken'] * egg_price

    if label_map:
        grouped['label'] = grouped[group_col].map(label_map).fillna(grouped[group_col].astype(str))
    else:
        grouped['label'] = grouped[group_col].astype(str)

    grouped['entity_type'] = group_col.replace("_id", "")
    return grouped

# === Charting ===
def plot_top_rcas(df: pd.DataFrame, outpath: str, top_n: int = 5, title: str = "Top Root Cause Contributors"):
    top = df.sort_values('rcas_normalized', ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(top['label'], top['rcas_normalized'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("RCAS (normalized)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# === RCA Analysis Runner ===
def run_rca_analysis(deliveries: pd.DataFrame, drivers: pd.DataFrame, by_batch: pd.DataFrame):
    report_dir = "data/RCA_Report"
    chart_dir = "charts/RCA"
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    # Load ANOVA filtering
    anova_csv_path = "data/ANOVA_Report/ANOVA_Table.csv"
    valid_factors = get_significant_anova_factors(anova_csv_path)

    driver_map = dict(zip(drivers['driver_id'], drivers['name']))
    entity_sets = []

    if "driver_id" in valid_factors:
        rcas_driver = compute_rcas(deliveries, "driver_id", label_map=driver_map, egg_price=egg_price_inr)
        entity_sets.append(("driver", rcas_driver))

    if "truck_id" in valid_factors:
        rcas_truck = compute_rcas(deliveries, "truck_id", egg_price=egg_price_inr)
        entity_sets.append(("truck", rcas_truck))

    if "route_id" in valid_factors:
        rcas_route = compute_rcas(deliveries, "route_id", egg_price=egg_price_inr)
        entity_sets.append(("route", rcas_route))

    rcas_batch = compute_rcas(by_batch, "batch_id", mode="aggregated", egg_price=egg_price_inr)
    entity_sets.append(("batch", rcas_batch))

    for entity, df in entity_sets:
        df_sorted = df.sort_values("rcas_normalized", ascending=False)
        csv_path = os.path.join(report_dir, f"{entity}_rcas.csv")
        chart_path = os.path.join(chart_dir, f"{entity}_rcas_top5.png")
        df_sorted.to_csv(csv_path, index=False)
        plot_top_rcas(df_sorted, chart_path, top_n=5, title=f"Top 5 {entity.capitalize()} RCAS Contributors")

    all_rcas = pd.concat([df for _, df in entity_sets], ignore_index=True)
    all_rcas = all_rcas.sort_values("rcas_normalized", ascending=False)
    all_rcas.to_csv(os.path.join(report_dir, "RCA_Report.csv"), index=False)
    plot_top_rcas(all_rcas, os.path.join(chart_dir, "Top_RCAS_Contributors.png"), top_n=20)

    print("RCAS reports saved to:", report_dir)
    print("RCAS charts saved to:", chart_dir)

# === Main Execution ===
if __name__ == "__main__":
    from src.supabase_loader import load_all_tables
    from src.preprocessing import compute_breakage_rate

    tables = load_all_tables()
    deliveries = tables["deliveries"]
    drivers = tables["drivers"]
    delivery_batches = tables["delivery_batches"]

    deliveries = compute_breakage_rate(deliveries)

    db = delivery_batches.merge(deliveries[['delivery_id', 'total_eggs_loaded', 'total_eggs_broken']],
                                on='delivery_id', how='left')
    db['est_broken'] = db.apply(lambda r: (r['eggs_from_batch'] * r['total_eggs_broken'] / r['total_eggs_loaded'])
                                if r['total_eggs_loaded'] > 0 else 0, axis=1)
    by_batch = db.groupby("batch_id").agg(
        deliveries_count=("delivery_id", "nunique"),
        eggs_from_batch=("eggs_from_batch", "sum"),
        est_broken=("est_broken", "sum")
    ).reset_index()
    by_batch['total_eggs_loaded'] = by_batch['eggs_from_batch']
    by_batch['total_eggs_broken'] = by_batch['est_broken']

    run_rca_analysis(deliveries, drivers, by_batch)