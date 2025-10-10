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

    # ANOVA model including batch_id
    model = ols("breakage_rate ~ C(driver_id) + C(truck_id) + C(route_id) + C(batch_id)", data=deliveries).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Save ANOVA table
    anova_table.to_csv(os.path.join(output_dir, "ANOVA_Table.csv"))
    print("ANOVA table saved to:", output_dir)

    # Generate boxplots for each factor
    for col in ['driver_id', 'truck_id', 'route_id', 'batch_id']:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=deliveries, x=col, y='breakage_rate')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Breakage Rate by {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, f"{col}_boxplot.png"), dpi=150)
        plt.close()

    print("ANOVA charts saved to:", chart_dir)

if __name__ == "__main__":
    from src.supabase_loader import load_all_tables
    from src.preprocessing import compute_breakage_rate

    tables = load_all_tables()
    deliveries = tables["deliveries"]
    delivery_batches = tables["delivery_batches"]

    deliveries = compute_breakage_rate(deliveries)

    run_multiway_anova(deliveries, delivery_batches)