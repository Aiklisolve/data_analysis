import os,math
import pandas as pd
import subprocess
from datetime import datetime
from typing import List, Optional,Dict,Any
import re
import requests
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from dotenv import load_dotenv

from src.supabase_loader import supabase
from src.route_optimizer import RouteOptimizer
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from urllib.parse import urljoin
from fastapi import Request
from urllib.parse import urljoin
from fastapi import Request
# === Load environment ===
load_dotenv()

# === Shared Config ===
DEPOT_LAT = float(os.getenv("DEPOT_LATITUDE", 17.3850))
DEPOT_LON = float(os.getenv("DEPOT_LONGITUDE", 78.4867))
FUEL_COST_PER_LITER = float(os.getenv("FUEL_COST_PER_LITER", 100.0))
FUEL_EFFICIENCY_KM_PER_LITER = float(os.getenv("FUEL_EFFICIENCY_KM_PER_LITER", 8.0))
ROAD_CORRECTION = float(os.getenv("ROAD_CORRECTION_FACTOR", 1.3))
PRICE_PER_EGG = float(os.getenv("PRICE_PER_EGG", 5.0))

# Project-relative paths
BASE_DIR = Path(__file__).resolve().parent        # ...\DataAnalysis\src
PROJECT_ROOT = BASE_DIR.parent                    # ...\DataAnalysis
CHARTS_DIR = PROJECT_ROOT / "charts"              # expect charts/RCA/*.png here

print(f"[DEBUG] __file__={__file__}")
print(f"[DEBUG] BASE_DIR={BASE_DIR}")
print(f"[DEBUG] CWD={os.getcwd()}")
print(f"[DEBUG] CHARTS_DIR={CHARTS_DIR} (exists={CHARTS_DIR.exists()})")

RCA_DIR = "data/RCA_Report"
RCA_CHART_DIR = "charts/RCA"
COST_DIR = "data/processed/cost_impact"
COST_CHART_DIR = "charts/cost_impact"
DELIVERY_CSV = "data/processed/deliveries_processed.csv"

optimizer = RouteOptimizer(road_correction_factor=ROAD_CORRECTION)



# === FastAPI App ===
app = FastAPI(title="Egg Analytics & Routing API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount /charts => <project>/charts
if CHARTS_DIR.exists():
    app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
    print(f"[OK] Serving /charts from: {CHARTS_DIR}")
else:
    print(f"[WARN] charts dir not found: {CHARTS_DIR}")

    
# === Analysis Trigger ===
def run_analysis_scripts():
    # run from project root so `-m src.*` imports work regardless of where uvicorn is started
    subprocess.run(["python", "-m", "src.root_cause"], check=True, cwd=str(PROJECT_ROOT))
    subprocess.run(["python", "-m", "src.cost_impact"], check=True, cwd=str(PROJECT_ROOT))


# === CSV Loader ===
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
VOLUME_M3 = {"loose_tray": 0.006, "packed_box": 0.03, "oil_tin": 0.02}
TRUCK_COST_PER_DAY_INR = 650.0

# === Dashboard Payload Builder ===
def get_dashboard_payload(z_threshold=2.0, cost_threshold=10000):
    run_analysis_scripts()

    deliveries = load_csv(DELIVERY_CSV)
    deliveries['date'] = pd.to_datetime(deliveries['date'], errors='coerce')
    deliveries = deliveries.dropna(subset=['date'])

    total_deliveries = len(deliveries)
    total_eggs_loaded = deliveries['total_eggs_loaded'].sum()
    total_eggs_broken = deliveries['total_eggs_broken'].sum()
    avg_breakage_rate = total_eggs_broken / total_eggs_loaded if total_eggs_loaded > 0 else 0
    total_cost_impact_inr = total_eggs_broken * PRICE_PER_EGG

    date_range_start = deliveries['date'].min().strftime("%Y-%m-%d") if total_deliveries else None
    date_range_end   = deliveries['date'].max().strftime("%Y-%m-%d") if total_deliveries else None

    problematic_entities = {}
    for entity in ["driver", "truck", "route", "batch"]:
        rcas_chart = _webpath(f"{RCA_CHART_DIR}/{entity}_rcas_top5.png")
        cost_chart = _webpath(f"{COST_CHART_DIR}/cost_by_{entity}.png")
        problematic_entities[entity + "s"] = {"rcas_chart": rcas_chart, "cost_chart": cost_chart}

    top_rcas_chart = _webpath(f"{RCA_CHART_DIR}/Top_RCAS_Contributors.png")

    return {
        "total_deliveries": total_deliveries,
        "total_eggs_loaded": int(total_eggs_loaded),
        "total_eggs_broken": int(total_eggs_broken),
        "avg_breakage_rate": round(avg_breakage_rate, 4),
        "total_cost_impact_inr": int(total_cost_impact_inr),
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "problematic_entities": problematic_entities,
        "top_rcas_chart": top_rcas_chart,
        # Optional: include server-side debug of where files are served from
        "__debug": {"charts_dir": str(CHARTS_DIR) if CHARTS_DIR else None},
    }

_WIN_ABS = re.compile(r"^[a-zA-Z]:[\\/]")
def is_windows_abs(p: str) -> bool:
    return bool(_WIN_ABS.match(p or ""))

def _webpath(p: str) -> str:
    r"""Make a filesystem-ish path safe for the web (/ not \, no leading ./)."""
    return (p or "").replace("\\", "/").lstrip("./")


def fs_to_web(p: str) -> str:
    if not p:
        return ""
    p = _webpath(p)
    if p.startswith("charts/"):
        p = p[len("charts/"):]
    return "/charts/" + p

def _supa_headers():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY not set in env")
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Accept": "application/json",
        # Uncomment if your tables live in a non-public schema
        # "Accept-Profile": "public",
        # "Content-Profile": "public",
    }

def supa_get(table: str, params: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.get(url, headers=_supa_headers(), params=params, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()

def supa_get(table: str, params: dict):
    """Simple GET wrapper for PostgREST."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params, headers=_supa_headers())
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        return r.json()

 
def to_abs_url(request: Request, web_path: str) -> str:
    if not web_path:
        return web_path
    if web_path.startswith("http://") or web_path.startswith("https://"):
        return web_path
    # ensure leading slash for urljoin to work as path-absolute
    wp = web_path if web_path.startswith("/") else "/" + web_path
    return urljoin(str(request.base_url), wp)



@app.get("/api/dashboard")
def dashboard(request: Request, z_threshold: float = 2.0, cost_threshold: float = 10000):
    payload = get_dashboard_payload(z_threshold=z_threshold, cost_threshold=cost_threshold)

    # Convert returned paths (which may be 'charts/...' or absolute 'D:\...') to web paths under /charts
    def map_entity(d: dict | None) -> dict | None:
        if not d: return d
        r = dict(d)
        r["rcas_chart"] = fs_to_web(r.get("rcas_chart", ""))
        r["cost_chart"] = fs_to_web(r.get("cost_chart", ""))
        # also include absolute URLs for convenience
        r["rcas_chart_url"] = to_abs_url(request, r["rcas_chart"])
        r["cost_chart_url"] = to_abs_url(request, r["cost_chart"])
        return r

    pe = payload.get("problematic_entities", {})
    for k in ["drivers", "trucks", "routes", "batchs"]:
        if k in pe:
            pe[k] = map_entity(pe[k])

    # Top RCAs
    payload["top_rcas_chart"] = fs_to_web(payload.get("top_rcas_chart", ""))
    payload["top_rcas_chart_url"] = to_abs_url(request, payload["top_rcas_chart"])

    return payload

@app.get("/debug/paths")
def debug_paths():
    return {
        "base_dir": str(BASE_DIR),
        "cwd": os.getcwd(),
        "charts_dir": str(CHARTS_DIR) if CHARTS_DIR else None,
        "exists": bool(CHARTS_DIR and CHARTS_DIR.exists()),
    }


# === Health Check ===
@app.get("/health")
async def health():
    return {"status": "ok"}

# === Route Optimization Models ===
class RoutingRequest(BaseModel):
    order_date: str
    num_trucks: int = 3

class StopOut(BaseModel):
    customer_id: str
    customer_name: Optional[str]
    latitude: float
    longitude: float
    distance_from_previous_km: Optional[float] = None

class TruckRouteOut(BaseModel):
    truck_id: str
    stops: List[StopOut]
    total_distance_km: float
    estimated_time_hours: Optional[float] = None
    fuel_liters: Optional[float] = None
    fuel_cost_inr: Optional[float] = None

class RoutingResult(BaseModel):
    routes: List[TruckRouteOut]
    total_distance_km: float
    baseline_distance_km: float
    distance_saved_km: float
    distance_saved_percent: float
    fuel_saved_liters: float
    cost_saved_inr: float

class LoadingPlan(BaseModel):
    truck_id: str
    layered_instructions: List[str]
    total_volume_used_m3: float
    utilization_percent: float

class PackingResult(BaseModel):
    loading_plans: List[LoadingPlan]
    trucks_used: int
    trucks_saved: int
    baseline_trucks: int
    avg_utilization_percent: float
    baseline_utilization_percent: float
    cost_saved_per_day_inr: float

class PackingRequest(BaseModel):
    order_date: str = Field(..., description="YYYY-MM-DD")

class DailyPlanRequest(BaseModel):
    order_date: str = Field(..., description="YYYY-MM-DD")
    num_trucks: int = Field(5, ge=1)



# === Supabase Order Fetcher ===
def fetch_orders_for_date_from_supabase(order_date: str):
    try:
        response = supabase.table("orders").select("*").eq("order_date", order_date).execute()
        df_orders = pd.DataFrame(response.data)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch 'orders' from Supabase: {e}")

    if df_orders.empty:
        return []

    try:
        customers_resp = supabase.table("customers").select("*").execute()
        df_customers = pd.DataFrame(customers_resp.data)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch 'customers' from Supabase: {e}")

    if not df_customers.empty and "customer_id" in df_customers.columns:
        merged = df_orders.merge(df_customers, on="customer_id", how="left", suffixes=("_order", "_cust"))
    else:
        merged = df_orders.copy()

    rows = []
    for _, r in merged.iterrows():
        lat = r.get("latitude") or r.get("latitude_cust")
        lon = r.get("longitude") or r.get("longitude_cust")
        if pd.isna(lat) or pd.isna(lon):
            continue
        rows.append({
            "order_id": str(r.get("order_id")),
            "customer_id": str(r.get("customer_id")),
            "customer_name": r.get("name") or r.get("customer_name"),
            "latitude": float(lat),
            "longitude": float(lon),
            "loose_trays": int(r.get("loose_trays")) if pd.notna(r.get("loose_trays")) else None,
            "packed_boxes": int(r.get("packed_boxes")) if pd.notna(r.get("packed_boxes")) else None,
            "oil_tins": int(r.get("oil_tins")) if pd.notna(r.get("oil_tins")) else None,
        })
    return rows

# === Route Optimization Endpoint ===
@app.post("/api/routing/optimize", response_model=RoutingResult, tags=["routing"])
async def optimize_routes(body: RoutingRequest):
    order_date = body.order_date
    num_trucks = max(1, int(body.num_trucks))

    try:
        rows = fetch_orders_for_date_from_supabase(order_date)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        raise HTTPException(status_code=404, detail=f"No orders found for {order_date}")

    depot = {
        "customer_id": "DEPOT",
        "customer_name": "Distribution Center",
        "latitude": DEPOT_LAT,
        "longitude": DEPOT_LON,
    }

    locations = [depot] + rows

    try:
        result = optimizer.optimize_multi_truck(locations, num_trucks=num_trucks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

    dist_mat = optimizer.create_distance_matrix(locations)
    baseline_distance = sum(dist_mat[0, i] * 2.0 for i in range(1, len(locations)))

    total_distance = float(result.get("total_distance_km", 0.0))
    distance_saved_km = max(0.0, baseline_distance - total_distance)
    distance_saved_percent = (distance_saved_km / baseline_distance * 100.0) if baseline_distance > 0 else 0.0

    baseline_fuel = baseline_distance / FUEL_EFFICIENCY_KM_PER_LITER
    baseline_cost = baseline_fuel * FUEL_COST_PER_LITER
    opt_fuel = total_distance / FUEL_EFFICIENCY_KM_PER_LITER
    opt_cost = opt_fuel * FUEL_COST_PER_LITER

    fuel_saved_liters = max(0.0, baseline_fuel - opt_fuel)
    cost_saved_inr = max(0.0, baseline_cost - opt_cost)

    routes_out = []
    for r in result.get("routes", []):
        stops_out = []
        prev_lat, prev_lon = DEPOT_LAT, DEPOT_LON
        for s in r.get("stops", []):
            lat, lon = float(s["latitude"]), float(s["longitude"])
            dist_prev = optimizer.haversine_distance(prev_lat, prev_lon, lat, lon)
            stops_out.append({
                "customer_id": s.get("customer_id"),
                "customer_name": s.get("customer_name"),
                "latitude": lat,
                "longitude": lon,
                "distance_from_previous_km": round(dist_prev, 3)
            })
            prev_lat, prev_lon = lat, lon

        routes_out.append({
            "truck_id": r.get("truck_id"),
            "stops": stops_out,
            "total_distance_km": round(float(r.get("total_distance_km", 0.0)), 3),
            "estimated_time_hours": round(float(r.get("estimated_time_hours", 0.0)), 3) if r.get("estimated_time_hours") is not None else None,
            "fuel_liters": round(opt_fuel, 3),
            "fuel_cost_inr": round(opt_cost, 2)
        })

    return {
        "routes": routes_out,
        "total_distance_km": round(total_distance, 3),
        "baseline_distance_km": round(baseline_distance, 3),
        "distance_saved_km": round(distance_saved_km, 3),
        "distance_saved_percent": round(distance_saved_percent, 2),
        "fuel_saved_liters": round(fuel_saved_liters, 3),
        "cost_saved_inr": round(cost_saved_inr, 2),
    }
