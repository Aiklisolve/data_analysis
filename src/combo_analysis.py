import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Define output folders
DATA_OUTDIR = "data/processed/combo_data"
CHART_OUTDIR = "charts/combo_charts"
os.makedirs(DATA_OUTDIR, exist_ok=True)
os.makedirs(CHART_OUTDIR, exist_ok=True)

def combo_breakage(deliveries, col1, col2, drivers=None, top_n=20):
    combo = deliveries.groupby([col1, col2]).agg(
        deliveries_count=('delivery_id', 'count'),
        eggs_loaded=('total_eggs_loaded', 'sum'),
        eggs_broken=('total_eggs_broken', 'sum')
    ).reset_index()
    combo['breakage_rate'] = combo['eggs_broken'] / combo['eggs_loaded']
    combo = combo.sort_values('breakage_rate', ascending=False)

    # Map driver_id to name if applicable
    if drivers is not None and col1 == "driver_id":
        driver_map = dict(zip(drivers['driver_id'], drivers['name']))
        combo['label'] = combo[col1].map(driver_map).fillna(combo[col1].astype(str)) + " × " + combo[col2].astype(str)
    elif drivers is not None and col2 == "driver_id":
        driver_map = dict(zip(drivers['driver_id'], drivers['name']))
        combo['label'] = combo[col1].astype(str) + " × " + combo[col2].map(driver_map).fillna(combo[col2].astype(str))
    else:
        combo['label'] = combo[col1].astype(str) + " × " + combo[col2].astype(str)

    # Save CSV
    csv_path = os.path.join(DATA_OUTDIR, f"breakage_by_{col1}_{col2}.csv")
    combo.to_csv(csv_path, index=False)

    # Plot top combos
    top = combo.head(top_n)
    plt.figure(figsize=(12, 5))
    plt.bar(top['label'], top['breakage_rate'] * 100)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Breakage rate (%)')
    plt.title(f'Breakage Rate by {col1} × {col2}')
    plt.tight_layout()

    chart_path = os.path.join(CHART_OUTDIR, f"{col1}_{col2}_combo_breakage.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return combo

def run_combo_analysis(deliveries, drivers):
    combo_breakage(deliveries, "driver_id", "truck_id", drivers=drivers)
    combo_breakage(deliveries, "driver_id", "route_id", drivers=drivers)
    combo_breakage(deliveries, "route_id", "truck_id", drivers=drivers)

if __name__ == "__main__":
    from src.supabase_loader import load_all_tables
    tables = load_all_tables()
    deliveries = tables["deliveries"]
    drivers = tables["drivers"]
    run_combo_analysis(deliveries, drivers)