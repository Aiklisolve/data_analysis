from supabase import create_client, Client
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables
load_dotenv()

# Connect to Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_table(table_name: str, limit: int = None) -> pd.DataFrame:
    """
    Fetches a table from Supabase and returns it as a DataFrame.
    """
    try:
        query = supabase.table(table_name).select("*")
        if limit:
            query = query.limit(limit)
        response = query.execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        print(f"Failed to fetch '{table_name}': {e}")
        return pd.DataFrame()

def load_all_tables(limit: int = None) -> dict:
    """
    Loads all relevant tables and returns a dictionary of DataFrames.
    """
    table_names = [
        "batches",
        "customers",
        "deliveries",
        "delivery_batches",
        "drivers",
        "orders",
        "routes",
        "trucks"
    ]

    dataframes = {}
    for name in table_names:
        df = fetch_table(name, limit)
        if not df.empty:
            print(f"Loaded '{name}' with {len(df)} rows.")
        else:
            print(f"'{name}' is empty or failed to load.")
        dataframes[name] = df

    return dataframes