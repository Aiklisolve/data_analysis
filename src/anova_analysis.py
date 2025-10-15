import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_multiway_anova(deliveries: pd.DataFrame, delivery_batches: pd.DataFrame,
                       output_dir: str = "data/ANOVA_Report", chart_dir: str = "charts/ANOVA_charts"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    # Merge batch_id into deliveries
    deliveries = deliveries.merge(delivery_batches[['delivery_id', 'batch_id']], on='delivery_id', how='left')

    # Compute breakage rate
    deliveries['breakage_rate'] = deliveries['total_eggs_broken'] / deliveries['total_eggs_loaded']

    # Drop rows with missing breakage rate or group columns
    deliveries = deliveries.dropna(subset=['breakage_rate', 'driver_id', 'truck_id', 'route_id', 'batch_id'])

    # Log group sizes and missing values
    print("\n[INFO] Group cardinality and missing values:")
    for col in ['driver_id', 'truck_id', 'route_id', 'batch_id']:
        print(f"  {col}: {deliveries[col].nunique()} unique, {deliveries[col].isna().sum()} missing")

    # Fit ANOVA model
    model = ols("breakage_rate ~ C(driver_id) + C(truck_id) + C(route_id) + C(batch_id)", data=deliveries).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Save ANOVA table
    anova_path = os.path.join(output_dir, "ANOVA_Table.csv")
    anova_table.to_csv(anova_path)
    print(f"\n✅ ANOVA table saved to: {anova_path}")

    # Print p-values for quick inspection
    print("\n[ANOVA Results]")
    print(anova_table[['F', 'PR(>F)']])

    # Warn if only one factor is significant
    significant = anova_table[anova_table["PR(>F)"] < 0.05]
    if len(significant) <= 1:
        print("[WARN] Only one significant factor detected. Check sample sizes or variance.")

    # Generate boxplots for each factor
    for col in ['driver_id', 'truck_id', 'route_id', 'batch_id']:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=deliveries, x=col, y='breakage_rate')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Breakage Rate by {col}")
        plt.tight_layout()
        chart_path = os.path.join(chart_dir, f"{col}_boxplot.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"[Chart] Saved boxplot: {chart_path}")

    print("\n✅ ANOVA charts saved to:", chart_dir)

if __name__ == "__main__":
    from src.supabase_loader import load_all_tables
    from src.preprocessing import compute_breakage_rate

    tables = load_all_tables()
    deliveries = tables["deliveries"]
    delivery_batches = tables["delivery_batches"]

    deliveries = compute_breakage_rate(deliveries)

    run_multiway_anova(deliveries, delivery_batches)