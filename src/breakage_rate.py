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
BREAKAGE_CHART_DIR = os.path.join(CHART_DIR, "breakage_rate")
COST_CHART_DIR = os.path.join(CHART_DIR, "cost_impact")
BREAKAGE_DATA_DIR = "data/processed/breakage_rate"
COST_DATA_DIR = "data/processed/cost_impact"

# Ensure folders exist
for folder in [CHART_DIR, BREAKAGE_CHART_DIR, COST_CHART_DIR, BREAKAGE_DATA_DIR, COST_DATA_DIR]:
    os.makedirs(folder, exist_ok=True)

def fleet_summary(deliveries):
    total_loaded = deliveries['total_eggs_loaded'].sum()
    total_broken = deliveries['total_eggs_broken'].sum()
    return {
        'total_loaded': int(total_loaded),
        'total_broken': int(total_broken),
        'fleet_breakage_rate': (total_broken / total_loaded) if total_loaded > 0 else 0
    }

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

def plot_bar_with_baseline(df, id_col, value_col, baseline, title, fname, top_n=20):
    df_plot = df.sort_values(value_col, ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(df_plot[id_col].astype(str), df_plot[value_col] * 100)
    plt.axhline(baseline * 100, color='red', linestyle='--', label=f'Fleet avg {baseline*100:.2f}%')
    plt.ylabel('Breakage rate (%)')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
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

    fleet = fleet_summary(deliveries)
    baseline = fleet['fleet_breakage_rate']

    deliveries['date'] = pd.to_datetime(deliveries['date'])
    deliveries['year_month'] = deliveries['date'].dt.to_period('M').astype(str)
    monthly = deliveries.groupby("year_month").agg(
        eggs_loaded=("total_eggs_loaded", "sum"),
        eggs_broken=("total_eggs_broken", "sum")
    ).reset_index()
    monthly['breakage_rate'] = monthly['eggs_broken'] / monthly['eggs_loaded']
    monthly.to_csv("data/processed/monthly_breakage.csv", index=False)
    plot_bar(monthly, "year_month", "breakage_rate", "Monthly Breakage Rate (%)",
             os.path.join(BREAKAGE_CHART_DIR, "monthly_breakage.png"), ylabel="Breakage rate (%)", scale=100)

    def group_and_plot(by_col, label, name_map=None):
        df = deliveries.groupby(by_col).agg(
            deliveries_count=("delivery_id", "count"),
            eggs_loaded=("total_eggs_loaded", "sum"),
            eggs_broken=("total_eggs_broken", "sum")
        ).reset_index()
        df['breakage_rate'] = df['eggs_broken'] / df['eggs_loaded']
        if name_map:
            df[label] = df[by_col].map(name_map)
        else:
            df[label] = df[by_col]
        df.to_csv(os.path.join(BREAKAGE_DATA_DIR, f"{label}_breakage.csv"), index=False)
        plot_bar_with_baseline(df, label, 'breakage_rate', baseline,
                               f'Breakage Rate by {label.capitalize()}',
                               os.path.join(BREAKAGE_CHART_DIR, f'{label}_vs_breakage.png'), top_n=20)
        return df

    driver_map = dict(zip(drivers['driver_id'], drivers['name']))
    by_driver = group_and_plot("driver_id", "driver_name", name_map=driver_map)
    by_truck = group_and_plot("truck_id", "truck_id")
    by_route = group_and_plot("route_id", "route_id")

    db = delivery_batches.merge(deliveries[['delivery_id', 'total_eggs_loaded', 'total_eggs_broken']],
                                on='delivery_id', how='left')
    db['est_broken'] = db.apply(lambda r: (r['eggs_from_batch'] * r['total_eggs_broken'] / r['total_eggs_loaded'])
                                if r['total_eggs_loaded'] > 0 else 0, axis=1)
    by_batch = db.groupby("batch_id").agg(
        deliveries_count=("delivery_id", "nunique"),
        eggs_from_batch=("eggs_from_batch", "sum"),
        est_broken=("est_broken", "sum")
    ).reset_index()
    by_batch['estimated_breakage_rate'] = by_batch['est_broken'] / by_batch['eggs_from_batch']
    by_batch.to_csv(os.path.join(BREAKAGE_DATA_DIR, "batch_breakage.csv"), index=False)
    plot_bar_with_baseline(by_batch, 'batch_id', 'estimated_breakage_rate', baseline,
                           'Estimated Breakage Rate by Batch',
                           os.path.join(BREAKAGE_CHART_DIR, 'batch_vs_breakage.png'), top_n=20)

    PRICE_PER_EGG = float(os.getenv("PRICE_PER_EGG", 5.0))
    by_driver['cost_impact'] = by_driver['eggs_broken'] * PRICE_PER_EGG
    by_truck['cost_impact'] = by_truck['eggs_broken'] * PRICE_PER_EGG
    by_route['cost_impact'] = by_route['eggs_broken'] * PRICE_PER_EGG
    by_batch['cost_impact'] = by_batch['est_broken'] * PRICE_PER_EGG

    by_driver.to_csv(os.path.join(COST_DATA_DIR, "driver_cost_impact.csv"), index=False)
    by_truck.to_csv(os.path.join(COST_DATA_DIR, "truck_cost_impact.csv"), index=False)
    by_route.to_csv(os.path.join(COST_DATA_DIR, "route_cost_impact.csv"), index=False)
    by_batch.to_csv(os.path.join(COST_DATA_DIR, "batch_cost_impact.csv"), index=False)

    plot_bar(by_driver, 'driver_name', 'cost_impact', 'Cost Impact by Driver (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_driver.png'), top_n=20)
    plot_bar(by_truck, 'truck_id', 'cost_impact', 'Cost Impact by Truck (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_truck.png'), top_n=20)
    plot_bar(by_route, 'route_id', 'cost_impact', 'Cost Impact by Route (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_route.png'), top_n=20)
    plot_bar(by_batch, 'batch_id', 'cost_impact', 'Cost Impact by Batch (₹)',
             os.path.join(COST_CHART_DIR, 'cost_by_batch.png'), top_n=20)

    print("Charts saved to:")
    print("  Breakage charts:", BREAKAGE_CHART_DIR)
    print("  Cost charts:    ", COST_CHART_DIR)
    print("CSVs saved to:")
    print("  Breakage data:  ", BREAKAGE_DATA_DIR)
    print("  Cost impact:    ", COST_DATA_DIR)

if __name__ == "__main__":
    run_all()