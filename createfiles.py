from src.supabase_loader import load_all_tables

tables = load_all_tables()
deliveries_df = tables["deliveries"]
routes_df = tables["routes"]
print(deliveries_df.head())
print(routes_df.head())