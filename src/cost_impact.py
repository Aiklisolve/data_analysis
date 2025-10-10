import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from src.supabase_loader import load_all_tables
from src.preprocessing import compute_breakage_rate, save_processed

# Load environment variables
load_dotenv()

plt.rcParams.update({'figure.autolayout': True})

# Output folders
CHART_DIR = "charts"
COST_CHART_DIR = os.path.join(CHART_DIR, "cost_impact")
COST_DATA_DIR = "data/processed/cost_impact"

# Ensure folders exist
for folder in [CHART_DIR, COST_CHART_DIR, COST_DATA_DIR]:
    os.makedirs(folder, exist_ok=True)

def plot_bar(df, id_col, value_col, title, fname, top_n=20, ylabel="Cost (₹)", scale=1.0):
    df_plot = df.sort_values(value_col, ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(df_plot[id_col].astype(str), df_plot[value_col] * scale)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def run_all():
    tables = load_all_tables()
    deliveries = tables["deliveries"]
    delivery_batches = tables["delivery_batches"]
    drivers = tables["drivers"]

    deliveries = compute_breakage_rate(deliveries)
    save_processed(deliveries, "deliveries_processed")

    PRICE_PER_EGG = float(os.getenv("PRICE_PER_EGG", 5.0))

    driver_map = dict(zip(drivers['driver_id'], drivers['name']))
    deliveries['driver_name'] = deliveries['driver_id'].map(driver_map)

    by_driver = deliveries.groupby("driver_name").agg(
        eggs_broken=("total_eggs_broken", "sum")
    ).reset_index()
    by_driver['cost_impact'] = by_driver['eggs_broken'] * PRICE_PER_EGG

    by_truck = deliveries.groupby("truck_id").agg(
        eggs_broken=("total_eggs_broken", "sum")
    ).reset_index()
    by_truck['cost_impact'] = by_truck['eggs_broken'] * PRICE_PER_EGG

    by_route = deliveries.groupby("route_id").agg(
        eggs_broken=("total_eggs_broken", "sum")
    ).reset_index()
    by_route['cost_impact'] = by_route['eggs_broken'] * PRICE_PER_EGG

    db = delivery_batches.merge(deliveries[['delivery_id', 'total_eggs_loaded', 'total_eggs_broken']],
                                on='delivery_id', how='left')
    db['est_broken'] = db.apply(lambda r: (r['eggs_from_batch'] * r['total_eggs_broken'] / r['total_eggs_loaded'])
                                if r['total_eggs_loaded'] > 0 else 0, axis=1)
    by_batch = db.groupby("batch_id").agg(
        est_broken=("est_broken", "sum")
    ).reset_index()
    by_batch['cost_impact'] = by_batch['est_broken'] * PRICE_PER_EGG

    # Save CSVs
    by_driver.to_csv(os.path.join(COST_DATA_DIR, "driver_cost_impact.csv"), index=False)
    by_truck.to_csv(os.path.join(COST_DATA_DIR, "truck_cost_impact.csv"), index=False)
    by_route.to_csv(os.path.join(COST_DATA_DIR, "route_cost_impact.csv"), index=False)
    by_batch.to_csv(os.path.join(COST_DATA_DIR, "batch_cost_impact.csv"), index=False)

    # Save charts
    plot_bar(by_driver, 'driver_name', 'cost_impact', 'Cost Impact by Driver (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_driver.png'), top_n=20)
    plot_bar(by_truck, 'truck_id', 'cost_impact', 'Cost Impact by Truck (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_truck.png'), top_n=20)
    plot_bar(by_route, 'route_id', 'cost_impact', 'Cost Impact by Route (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_route.png'), top_n=20)
    plot_bar(by_batch, 'batch_id', 'cost_impact', 'Cost Impact by Batch (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_batch.png'), top_n=20)

    print("Cost charts saved to:", COST_CHART_DIR)
    print("Cost impact CSVs saved to:", COST_DATA_DIR)

if __name__ == "__main__":
    run_all()