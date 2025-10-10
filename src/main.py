# main.py
"""
Complete Production FastAPI Server
Includes:
- Route Optimization (Module B) with full cost analysis
- Space Optimization (Module C) with all original models
- Integrated Optimization
- RCAS Dashboard
All from src/ folder
"""

import os
import math
import pandas as pd
import subprocess
from typing import List, Optional, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import re
import requests
import httpx
# === Imports ===
from src.supabase_loader import supabase, load_all_tables
from src.route_optimizer_v2 import RouteOptimizerV2
from src.space_optimizer import SpaceOptimizer
from src.integrated_optimizer import IntegratedOptimizer
from urllib.parse import urljoin
from fastapi import Request
from urllib.parse import urljoin
from fastapi import Request
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# === Environment ===
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent        # ...\DataAnalysis\src
PROJECT_ROOT = BASE_DIR.parent                    # ...\DataAnalysis
CHARTS_DIR = PROJECT_ROOT / "charts" 
# Depot
DEPOT_LAT = float(os.getenv("DEPOT_LATITUDE", "17.3850"))
DEPOT_LON = float(os.getenv("DEPOT_LONGITUDE", "78.4867"))

# Route config
FUEL_COST_PER_LITER = float(os.getenv("FUEL_COST_PER_LITER", "100.0"))
FUEL_EFFICIENCY_KM_PER_LITER = float(os.getenv("FUEL_EFFICIENCY_KM_PER_LITER", "8.0"))
ROAD_CORRECTION = float(os.getenv("ROAD_CORRECTION_FACTOR", "1.3"))
AVG_SPEED_KMPH = float(os.getenv("AVG_SPEED_KMPH", "40.0"))
MAX_CUSTOMERS_PER_ROUTE = int(os.getenv("MAX_CUSTOMERS_PER_ROUTE", "15"))
MAX_AVAILABLE_TRUCKS = int(os.getenv("MAX_AVAILABLE_TRUCKS", "15"))
TRUCK_COST_PER_DAY = float(os.getenv("TRUCK_COST_PER_DAY", "650.0"))

# HLD baseline
HLD_BASELINE_TRUCKS = 5
HLD_AVG_KM_PER_TRUCK = 120.0
HLD_BASELINE_TOTAL_KM = HLD_BASELINE_TRUCKS * HLD_AVG_KM_PER_TRUCK

# RCAS & Space config
PRICE_PER_EGG = float(os.getenv("PRICE_PER_EGG", "5.0"))
DEFAULT_BASELINE_TRUCKS = int(os.getenv("DEFAULT_BASELINE_TRUCKS", "15"))
RCA_CHART_DIR = "charts/RCA"
COST_CHART_DIR = "charts/cost_impact"
DELIVERY_CSV = "data/processed/deliveries_processed.csv"

# === App ===
app = FastAPI(
    title="Aiklisolve Integrated Distribution Optimization API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount /charts => <project>/charts
if CHARTS_DIR.exists():
    app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
    print(f"[OK] Serving /charts from: {CHARTS_DIR}")
else:
    print(f"[WARN] charts dir not found: {CHARTS_DIR}")


# === Initialize Optimizers ===
route_optimizer = RouteOptimizerV2(
    road_correction_factor=ROAD_CORRECTION,
    fuel_cost_per_liter=FUEL_COST_PER_LITER,
    fuel_efficiency_km_per_liter=FUEL_EFFICIENCY_KM_PER_LITER,
    avg_speed_kmph=AVG_SPEED_KMPH,
    max_customers_per_route=MAX_CUSTOMERS_PER_ROUTE,
    max_available_trucks=MAX_AVAILABLE_TRUCKS
)

space_optimizer = SpaceOptimizer()
integrated_optimizer = IntegratedOptimizer(route_optimizer, space_optimizer)

print("[INFO] ✅ All optimizers initialized from src/")


# =====================================================================
# PYDANTIC MODELS - Route Optimization
# =====================================================================

class RoutingRequest(BaseModel):
    order_date: str
    max_customers_per_route: Optional[int] = None


class StopOut(BaseModel):
    customer_id: str
    customer_name: Optional[str] = None
    latitude: float
    longitude: float
    distance_from_previous_km: Optional[float] = None
    merged_customers: Optional[List[str]] = None


class TruckRouteOut(BaseModel):
    truck_id: str
    stops: List[StopOut]
    total_distance_km: float
    estimated_time_hours: Optional[float] = None
    fuel_liters: Optional[float] = None
    fuel_cost_inr: Optional[float] = None
    num_stops: Optional[int] = None


class RoutingResult(BaseModel):
    routes: List[TruckRouteOut]
    total_distance_km: float
    baseline_distance_km: float
    distance_saved_km: float
    distance_saved_percent: float
    fuel_saved_liters: float
    fuel_cost_saved_inr: float
    truck_cost_saved_inr: float
    total_cost_saved_inr: float
    num_trucks_used: int
    trucks_saved: int


# =====================================================================
# PYDANTIC MODELS - Space Optimization (YOUR ORIGINAL MODELS)
# =====================================================================

class LayerOrder(BaseModel):
    order_id: str
    customer_id: Optional[str] = None
    volume_m3: float
    weight_kg: float
    dims: Optional[List[float]] = Field(default_factory=list)
    oil_tins: int = 0
    packed_boxes: int = 0
    loose_trays: int = 0
    fragile: bool = False


class TruckLayers(BaseModel):
    bottom: List[LayerOrder] = Field(default_factory=list)
    middle: List[LayerOrder] = Field(default_factory=list)
    top: List[LayerOrder] = Field(default_factory=list)


class LoadingPlan(BaseModel):
    truck_id: str
    truck_category: Optional[str] = None
    layers: TruckLayers
    height_used_cm: Optional[float] = 0.0
    total_capacity_m3: float
    total_volume_used_m3: float
    utilization_percent: float


class UnassignedOrder(BaseModel):
    order_id: str
    customer_id: Optional[str] = None
    volume_cm3: float = 0.0
    weight_kg: float = 0.0
    reason: Optional[str] = None


class SpaceRequest(BaseModel):
    order_date: str


class SpaceResponse(BaseModel):
    loading_plans: List[LoadingPlan]
    trucks_used: int
    trucks_saved: int
    baseline_trucks: int
    avg_utilization_percent: float
    baseline_utilization_percent: float
    cost_saved_per_day_inr: float
    unassigned_orders: List[UnassignedOrder] = Field(default_factory=list)


# =====================================================================
# RCAS DASHBOARD (Ali's code - UNCHANGED)
# =====================================================================

def run_analysis_scripts():
    try:
        # subprocess.run(["python", "-m", "src.root_cause"], check=True)
        # subprocess.run(["python", "-m", "src.cost_impact"], check=True)
          subprocess.run(["python", "-m", "src.root_cause"], check=True, cwd=str(PROJECT_ROOT))
          subprocess.run(["python", "-m", "src.cost_impact"], check=True, cwd=str(PROJECT_ROOT))

    except Exception as e:
        print(f"[WARNING] RCAS analysis scripts failed: {e}")


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
VOLUME_M3 = {"loose_tray": 0.006, "packed_box": 0.03, "oil_tin": 0.02}
TRUCK_COST_PER_DAY_INR = 650.0

@app.get("/api/dashboard", tags=["rcas"])
def dashboard(z_threshold: float = 2.0, cost_threshold: float = 10000):
    run_analysis_scripts()
    deliveries = load_csv(DELIVERY_CSV)
    
    if deliveries.empty:
        raise HTTPException(status_code=404, detail="No delivery data found")
    
    deliveries['date'] = pd.to_datetime(deliveries['date'], errors='coerce')
    deliveries = deliveries.dropna(subset=['date'])

    total_deliveries = len(deliveries)
    total_eggs_loaded = deliveries['total_eggs_loaded'].sum()
    total_eggs_broken = deliveries['total_eggs_broken'].sum()
    avg_breakage_rate = total_eggs_broken / total_eggs_loaded if total_eggs_loaded > 0 else 0
    total_cost_impact_inr = total_eggs_broken * PRICE_PER_EGG

    date_range_start = deliveries['date'].min().strftime("%Y-%m-%d")
    date_range_end = deliveries['date'].max().strftime("%Y-%m-%d")

    problematic_entities = {}
    for entity in ["driver", "truck", "route", "batch"]:
        rcas_chart = os.path.join(RCA_CHART_DIR, f"{entity}_rcas_top5.png")
        cost_chart = os.path.join(COST_CHART_DIR, f"cost_by_{entity}.png")
        problematic_entities[entity + "s"] = {
            "rcas_chart": rcas_chart,
            "cost_chart": cost_chart
        }

    top_rcas_chart = os.path.join(RCA_CHART_DIR, "Top_RCAS_Contributors.png")

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


# =====================================================================
# HELPER FUNCTIONS (YOUR ORIGINAL LOGIC)
# =====================================================================

def fetch_orders_for_date_from_supabase(order_date: str) -> List[Dict]:
    """YOUR ORIGINAL FUNCTION - kept as-is"""
    try:
        response = supabase.table("orders").select("*").eq("order_date", order_date).execute()
        df_orders = pd.DataFrame(response.data)
        print(f"[debug] Supabase fetched {len(df_orders)} orders for {order_date}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch 'orders' from Supabase: {e}")

    if df_orders.empty:
        print(f"[debug] ❌ No orders found for {order_date}")
        return []

    try:
        customers_resp = supabase.table("customers").select("*").execute()
        df_customers = pd.DataFrame(customers_resp.data)
        print(f"[debug] Supabase fetched {len(df_customers)} customers")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch 'customers' from Supabase: {e}")

    if not df_customers.empty and "customer_id" in df_customers.columns:
        merged = df_orders.merge(
            df_customers, 
            on="customer_id", 
            how="left", 
            suffixes=("_order", "_cust")
        )
    else:
        merged = df_orders.copy()

    rows = []
    for _, r in merged.iterrows():
        lat = r.get("latitude") or r.get("latitude_cust")
        lon = r.get("longitude") or r.get("longitude_cust")
        
        if pd.isna(lat) or pd.isna(lon):
            print(f"[debug] Skipping order {r.get('order_id')} - missing coordinates")
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

    print(f"[debug] ✅ Returning {len(rows)} valid orders for {order_date}")
    return rows


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/", tags=["info"])
async def root():
    return {
        "service": "Aiklisolve Integrated API",
        "version": "3.0.0",
        "hld_baseline": {
            "trucks": HLD_BASELINE_TRUCKS,
            "km_per_truck": HLD_AVG_KM_PER_TRUCK,
            "total_km": HLD_BASELINE_TOTAL_KM,
            "truck_cost_per_day": TRUCK_COST_PER_DAY
        },
        "endpoints": {
            "routing": "/api/routing/optimize",
            "space": "/api/space/optimize",
            "integrated": "/api/integrated/optimize",
            "rcas": "/dashboard",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Aiklisolve API",
        "version": "3.0.0",
        "hld_compliant": True,
        "cost_tracking": "fuel + trucks"
    }


@app.post("/api/routing/optimize", response_model=RoutingResult, tags=["routing"])
async def optimize_routes(body: RoutingRequest):
    """
    Route Optimization with complete cost analysis
    """
    order_date = body.order_date
    max_customers = body.max_customers_per_route or MAX_CUSTOMERS_PER_ROUTE

    print(f"\n{'='*70}")
    print(f"[OPTIMIZE] Date: {order_date}")
    print(f"[OPTIMIZE] Max customers/route: {max_customers}")
    print(f"[OPTIMIZE] Fleet size limit: {MAX_AVAILABLE_TRUCKS} trucks")
    print(f"{'='*70}")

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
    n_original_orders = len(rows)

    print(f"[debug] Total orders fetched: {n_original_orders}")

    merged_locations, customer_mapping = route_optimizer.merge_colocated_customers(locations)
    n_unique_locations = len(merged_locations) - 1
    
    print(f"[debug] Unique delivery locations: {n_unique_locations}")
    print(f"[debug] Duplicates merged: {n_original_orders - n_unique_locations}")

    try:
        result = route_optimizer.optimize_routes(locations)
        print(f"[debug] ✅ Optimization complete: {result['num_trucks_used']} trucks used")
    except Exception as e:
        print(f"[ERROR] Optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

    total_distance = float(result.get("total_distance_km", 0.0))
    trucks_used = result.get("num_trucks_used", 0)
    
    # === COMPLETE COST ANALYSIS ===
    print(f"\n[debug] === HLD Baseline (DS-O-004) ===")
    print(f"  Current practice: {HLD_BASELINE_TRUCKS} trucks/day")
    print(f"  Average distance: {HLD_AVG_KM_PER_TRUCK} km/truck")
    print(f"  Total baseline: {HLD_BASELINE_TOTAL_KM} km")

    baseline_distance = HLD_BASELINE_TOTAL_KM
    trucks_saved = HLD_BASELINE_TRUCKS - trucks_used

    distance_saved_km = max(0.0, baseline_distance - total_distance)
    distance_saved_percent = (
        (distance_saved_km / baseline_distance * 100.0) 
        if baseline_distance > 0 else 0.0
    )

    baseline_fuel = baseline_distance / FUEL_EFFICIENCY_KM_PER_LITER
    baseline_fuel_cost = baseline_fuel * FUEL_COST_PER_LITER
    opt_fuel = total_distance / FUEL_EFFICIENCY_KM_PER_LITER
    opt_fuel_cost = opt_fuel * FUEL_COST_PER_LITER
    fuel_saved_liters = max(0.0, baseline_fuel - opt_fuel)
    fuel_cost_saved = max(0.0, baseline_fuel_cost - opt_fuel_cost)
    
    baseline_truck_cost = HLD_BASELINE_TRUCKS * TRUCK_COST_PER_DAY
    opt_truck_cost = trucks_used * TRUCK_COST_PER_DAY
    truck_cost_saved = max(0.0, baseline_truck_cost - opt_truck_cost)
    
    total_cost_saved = fuel_cost_saved + truck_cost_saved
    
    print(f"\n[debug] === Complete Cost Analysis ===")
    print(f"  SAVINGS:")
    print(f"    - Fuel: ₹{fuel_cost_saved:.2f}")
    print(f"    - Trucks: ₹{truck_cost_saved:.2f}")
    print(f"    - TOTAL: ₹{total_cost_saved:.2f}")
    print(f"{'='*70}\n")

    # === FORMAT OUTPUT ===
    routes_out = []
    
    for r in result.get("routes", []):
        stops_out = []
        prev_lat, prev_lon = DEPOT_LAT, DEPOT_LON
        
        for s in r.get("stops", []):
            lat = float(s["latitude"])
            lon = float(s["longitude"])
            dist_prev = route_optimizer.haversine_distance(prev_lat, prev_lon, lat, lon)
            
            stop_dict = {
                "customer_id": s.get("customer_id"),
                "customer_name": s.get("customer_name"),
                "latitude": lat,
                "longitude": lon,
                "distance_from_previous_km": round(dist_prev, 3)
            }
            
            if 'merged_customers' in s and len(s['merged_customers']) > 1:
                stop_dict['merged_customers'] = s['merged_customers']
            
            stops_out.append(StopOut(**stop_dict))
            prev_lat, prev_lon = lat, lon

        routes_out.append(TruckRouteOut(
            truck_id=r.get("truck_id"),
            stops=stops_out,
            total_distance_km=round(float(r.get("total_distance_km", 0.0)), 3),
            estimated_time_hours=round(float(r.get("estimated_time_hours", 0.0)), 3),
            fuel_liters=round(float(r.get("fuel_liters", 0.0)), 3),
            fuel_cost_inr=round(float(r.get("fuel_cost_inr", 0.0)), 2),
            num_stops=r.get("num_stops", len(stops_out))
        ))

    return RoutingResult(
        routes=routes_out,
        total_distance_km=round(total_distance, 3),
        baseline_distance_km=round(baseline_distance, 3),
        distance_saved_km=round(distance_saved_km, 3),
        distance_saved_percent=round(distance_saved_percent, 2),
        fuel_saved_liters=round(fuel_saved_liters, 3),
        fuel_cost_saved_inr=round(fuel_cost_saved, 2),
        truck_cost_saved_inr=round(truck_cost_saved, 2),
        total_cost_saved_inr=round(total_cost_saved, 2),
        num_trucks_used=trucks_used,
        trucks_saved=trucks_saved
    )


@app.post("/api/space/optimize", response_model=SpaceResponse, tags=["space"])
async def optimize_space(body: SpaceRequest):
    """
    Space Optimization with YOUR ORIGINAL LOGIC
    """
    try:
        target_date = pd.to_datetime(body.order_date).strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid order_date format.")

    # Query Supabase for matching orders
    response = supabase.table("orders").select("*").eq("order_date", target_date).execute()
    rows = response.data
    df_orders = pd.DataFrame(rows)

    print(f"[space] Orders retrieved from Supabase: {len(df_orders)}")

    def safe_int(val):
        try:
            return int(float(val))
        except Exception:
            return 0

    orders = []
    for _, row in df_orders.iterrows():
        orders.append({
            "order_id": str(row.get("order_id") or row.get("id") or ""),
            "customer_id": str(row.get("customer_id") or ""),
            "oil_tins": safe_int(row.get("oil_tins", 0)),
            "packed_boxes": safe_int(row.get("packed_boxes", 0)),
            "loose_trays": safe_int(row.get("loose_trays", 0)),
        })

    print(f"[space] Parsed orders: {len(orders)}")

    try:
        result = space_optimizer.optimize(orders)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Space optimizer failed: {e}")

    print(f"[space] Optimizer result: {result.get('trucks_used')} trucks used")

    try:
        loading_plans = []
        for lp in result.get("loading_plans", []):
            layers_raw = lp.get("layers", {})
            bottom = [LayerOrder(**o) for o in layers_raw.get("bottom", [])]
            middle = [LayerOrder(**o) for o in layers_raw.get("middle", [])]
            top = [LayerOrder(**o) for o in layers_raw.get("top", [])]
            loading_plans.append(LoadingPlan(
                truck_id=lp.get("truck_id"),
                truck_category=lp.get("truck_category"),
                layers=TruckLayers(bottom=bottom, middle=middle, top=top),
                height_used_cm=lp.get("height_used_cm", 0.0),
                total_capacity_m3=lp.get("total_capacity_m3", 0.0),
                total_volume_used_m3=lp.get("total_volume_used_m3", 0.0),
                utilization_percent=lp.get("utilization_percent", 0.0)
            ))

        unassigned = [UnassignedOrder(
            order_id=u.get("order_id"),
            customer_id=u.get("customer_id"),
            volume_cm3=u.get("volume_cm3", 0.0),
            weight_kg=u.get("weight_kg", 0.0),
            reason=u.get("reason")
        ) for u in result.get("unassigned_orders", [])]

        return SpaceResponse(
            loading_plans=loading_plans,
            trucks_used=result.get("trucks_used", 0),
            trucks_saved=result.get("trucks_saved", 0),
            baseline_trucks=result.get("baseline_trucks", DEFAULT_BASELINE_TRUCKS),
            avg_utilization_percent=result.get("avg_utilization_percent", 0.0),
            baseline_utilization_percent=result.get("baseline_utilization_percent", 0.0),
            cost_saved_per_day_inr=result.get("cost_saved_per_day_inr", 0.0),
            unassigned_orders=unassigned
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate optimizer output: {e}")


@app.post("/api/integrated/optimize", tags=["integrated"])
async def optimize_integrated(body: RoutingRequest):
    """Integrated optimization endpoint"""
    order_date = body.order_date
    
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
        "longitude": DEPOT_LON
    }
    
    locations = [depot] + rows
    
    try:
        result = integrated_optimizer.optimize_integrated(
            locations=locations,
            orders=rows
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integrated optimization error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
