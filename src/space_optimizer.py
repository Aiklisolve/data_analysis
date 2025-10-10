from typing import List, Dict, Tuple
from copy import deepcopy
from itertools import combinations
from collections import defaultdict

# ---------------------------------------------------------------------
# Product specifications (dimensions in meters)
# ---------------------------------------------------------------------
PRODUCT_SPECS = {
    "oil_tin": {"length": 0.3, "width": 0.25, "height": 0.3, "weight": 15},
    "packed_box": {"length": 0.4, "width": 0.3, "height": 0.25, "weight": 3},
    "loose_tray": {"length": 0.5, "width": 0.4, "height": 0.15, "weight": 1},
}

# ---------------------------------------------------------------------
# Truck specifications (dimensions in meters)
# ---------------------------------------------------------------------
TRUCK_SPECS = {
    "small":  {"count": 5, "length": 3.0, "width": 2.0, "height": 2.0, "volume": 12.0},
    "medium": {"count": 7, "length": 4.5, "width": 2.2, "height": 2.5, "volume": 24.75},
    "large":  {"count": 3, "length": 6.0, "width": 2.5, "height": 2.8, "volume": 42.0},
}

class SpaceOptimizer:
    def __init__(self):
        self.trucks = []
        self.truck_counter = 0

    def _initialize_fleet_top_down(self, orders: List[Dict]):
        total_volume = sum(o["volume_m3"] for o in orders)

        def all_orders_fit(spec):
            return all(o["volume_m3"] <= spec["volume"] and o["stacking_height"] <= spec["height"] for o in orders)

        sorted_categories = sorted(TRUCK_SPECS.items(), key=lambda x: -x[1]["volume"])

        for category, spec in sorted_categories:
            utilization = total_volume / spec["volume"]
            if utilization <= 0.9 and all_orders_fit(spec):
                self.trucks = []
                for _ in range(spec["count"]):
                    self.truck_counter += 1
                    self.trucks.append({
                        "truck_id": f"T{self.truck_counter:02d}",
                        "truck_category": category,
                        "specs": spec,
                        "used_volume": 0.0,
                        "max_height_used": 0.0,
                        "layers": {"bottom": [], "middle": [], "top": []}
                    })
                return

        # fallback to mixed fleet
        self._initialize_fleet()

    def _initialize_fleet(self):
        truck_list = []
        for category, spec in TRUCK_SPECS.items():
            for _ in range(spec["count"]):
                self.truck_counter += 1
                truck_list.append({
                    "truck_id": f"T{self.truck_counter:02d}",
                    "truck_category": category,
                    "specs": spec,
                    "used_volume": 0.0,
                    "max_height_used": 0.0,
                    "layers": {"bottom": [], "middle": [], "top": []}
                })
        self.trucks = sorted(truck_list, key=lambda t: t["specs"]["volume"])

    def compute_order_properties(self, order: Dict) -> Dict:
        oil = order.get("oil_tins", 0)
        box = order.get("packed_boxes", 0)
        tray = order.get("loose_trays", 0)

        oil_volume = oil * (PRODUCT_SPECS["oil_tin"]["length"] *
                            PRODUCT_SPECS["oil_tin"]["width"] *
                            PRODUCT_SPECS["oil_tin"]["height"])
        box_volume = box * (PRODUCT_SPECS["packed_box"]["length"] *
                            PRODUCT_SPECS["packed_box"]["width"] *
                            PRODUCT_SPECS["packed_box"]["height"])
        tray_volume = tray * (PRODUCT_SPECS["loose_tray"]["length"] *
                              PRODUCT_SPECS["loose_tray"]["width"] *
                              PRODUCT_SPECS["loose_tray"]["height"])

        order["oil_volume_m3"] = oil_volume
        order["box_volume_m3"] = box_volume
        order["tray_volume_m3"] = tray_volume
        order["volume_m3"] = oil_volume + box_volume + tray_volume
        order["weight_kg"] = (
            oil * PRODUCT_SPECS["oil_tin"]["weight"] +
            box * PRODUCT_SPECS["packed_box"]["weight"] +
            tray * PRODUCT_SPECS["loose_tray"]["weight"]
        )
        order["fragile"] = tray > 0

        order_height = 0.0
        if oil > 0:
            order_height += PRODUCT_SPECS["oil_tin"]["height"]
        if box > 0:
            order_height += PRODUCT_SPECS["packed_box"]["height"]
        if tray > 0:
            order_height += PRODUCT_SPECS["loose_tray"]["height"]

        order["stacking_height"] = order_height
        return order

    def _can_fit_complete_order(self, truck: Dict, order: Dict) -> bool:
        specs = truck["specs"]
        remaining_volume = specs["volume"] - truck["used_volume"]
        remaining_height = specs["height"] - truck["max_height_used"]
        return order["volume_m3"] <= remaining_volume and order["stacking_height"] <= remaining_height

    def _pack_complete_order(self, truck: Dict, order: Dict):
        if order["oil_tins"] > 0:
            truck["layers"]["bottom"].append({
                "order_id": order["order_id"],
                "customer_id": order.get("customer_id"),
                "volume_m3": order["oil_volume_m3"],
                "weight_kg": order["oil_tins"] * PRODUCT_SPECS["oil_tin"]["weight"],
                "dims": [],
                "oil_tins": order["oil_tins"],
                "packed_boxes": 0,
                "loose_trays": 0,
                "fragile": False
            })
        if order["packed_boxes"] > 0:
            truck["layers"]["middle"].append({
                "order_id": order["order_id"],
                "customer_id": order.get("customer_id"),
                "volume_m3": order["box_volume_m3"],
                "weight_kg": order["packed_boxes"] * PRODUCT_SPECS["packed_box"]["weight"],
                "dims": [],
                "oil_tins": 0,
                "packed_boxes": order["packed_boxes"],
                "loose_trays": 0,
                "fragile": False
            })
        if order["loose_trays"] > 0:
            truck["layers"]["top"].append({
                "order_id": order["order_id"],
                "customer_id": order.get("customer_id"),
                "volume_m3": order["tray_volume_m3"],
                "weight_kg": order["loose_trays"] * PRODUCT_SPECS["loose_tray"]["weight"],
                "dims": [],
                "oil_tins": 0,
                "packed_boxes": 0,
                "loose_trays": order["loose_trays"],
                "fragile": True
            })

        truck["used_volume"] += order["volume_m3"]
        truck["max_height_used"] += order["stacking_height"]

    def assign_orders_to_trucks(self, orders: List[Dict]) -> Tuple[List[Dict], List[str]]:
        MIN_UTILIZATION = 50  # Minimum required utilization for batch assignment

        unassigned = orders.copy()
        locked_trucks = []
        open_trucks = deepcopy(self.trucks)

        def simulate_utilization(truck, batch):
            total_volume = sum(o["volume_m3"] for o in batch)
            total_height = sum(o["stacking_height"] for o in batch)
            max_volume = truck["specs"]["volume"] * 0.9
            fits = total_volume <= max_volume and total_height <= truck["specs"]["height"]
            utilization = (total_volume / truck["specs"]["volume"]) * 100 if fits else 0
            return fits, utilization

        def lock_if_full(truck):
            utilization = (truck["used_volume"] / truck["specs"]["volume"]) * 100
            if utilization >= 90:
                locked_trucks.append(truck)
                open_trucks.remove(truck)

        # Batch and assign orders
        while unassigned:
            best_batch = None
            best_truck = None
            best_utilization = 0.0

            for r in range(1, min(6, len(unassigned) + 1)):
                for batch in combinations(unassigned, r):
                    for truck in open_trucks:
                        fits, utilization = simulate_utilization(truck, batch)
                        if fits and utilization >= MIN_UTILIZATION and utilization > best_utilization:
                            projected_volume = truck["used_volume"] + sum(o["volume_m3"] for o in batch)
                            projected_height = truck["max_height_used"] + sum(o["stacking_height"] for o in batch)
                            if projected_volume <= truck["specs"]["volume"] * 0.9 and projected_height <= truck["specs"]["height"]:
                                best_batch = batch
                                best_truck = truck
                                best_utilization = utilization

            if best_batch and best_truck:
                for order in best_batch:
                    self._pack_complete_order(best_truck, order)
                    unassigned = [o for o in unassigned if o["order_id"] != order["order_id"]]
                lock_if_full(best_truck)
            else:
                break  # No viable batch found

        # Final fallback: try assigning remaining orders individually with utilization check
        for order in unassigned[:]:
            for truck in open_trucks:
                projected_volume = truck["used_volume"] + order["volume_m3"]
                projected_height = truck["max_height_used"] + order["stacking_height"]
                utilization = (order["volume_m3"] / truck["specs"]["volume"]) * 100
                if (
                    projected_volume <= truck["specs"]["volume"] * 0.9
                    and projected_height <= truck["specs"]["height"]
                    and utilization >= MIN_UTILIZATION
                ):
                    self._pack_complete_order(truck, order)
                    unassigned = [o for o in unassigned if o["order_id"] != order["order_id"]]
                    lock_if_full(truck)
                    break

        used_trucks = locked_trucks + [t for t in open_trucks if t["used_volume"] > 0]
        unassigned_ids = [o["order_id"] for o in unassigned]
        return used_trucks, unassigned_ids





    def group_orders_by_id(self, orders: List[Dict]) -> List[Dict]:
        grouped = defaultdict(list)
        for o in orders:
            grouped[o["order_id"]].append(o)

        merged_orders = []
        for group in grouped.values():
            merged = group[0].copy()
            merged["volume_m3"] = sum(o["volume_m3"] for o in group)
            merged["stacking_height"] = sum(o["stacking_height"] for o in group)
            merged["item_names"] = [o.get("item_name") for o in group]
            merged_orders.append(merged)

        return merged_orders
    
    def optimize(self, orders: List[Dict]) -> Dict:
        raw_orders = [self.compute_order_properties(o) for o in deepcopy(orders)]
        orders = self.group_orders_by_id(raw_orders)
        self._initialize_fleet()  # Mixed fleet for flexible assignment

        used_trucks, unassigned = self.assign_orders_to_trucks(orders)

        results = []
        for truck in used_trucks:
            specs = truck["specs"]
            utilization = (truck["used_volume"] / specs["volume"]) * 100
            results.append({
                "truck_id": truck["truck_id"],
                "truck_category": truck["truck_category"],
                "layers": truck["layers"],
                "height_used_cm": round(truck["max_height_used"] * 100, 2),
                "total_capacity_m3": specs["volume"],
                "total_volume_used_m3": round(truck["used_volume"], 2),
                "utilization_percent": round(utilization, 2)
            })

        baseline_trucks = sum(spec["count"] for spec in TRUCK_SPECS.values())
        trucks_used = len(results)
        trucks_saved = max(0, baseline_trucks - trucks_used)
        avg_utilization = (
            sum(r["utilization_percent"] for r in results) / trucks_used
            if trucks_used > 0 else 0
        )

        unassigned_details = []
        for order in orders:
            if order["order_id"] in unassigned:
                unassigned_details.append({
                    "order_id": order["order_id"],
                    "customer_id": order.get("customer_id"),
                    "volume_cm3": round(order["volume_m3"] * 1_000_000, 2),
                    "weight_kg": round(order["weight_kg"], 2),
                    "reason": "Exceeds remaining truck capacity"
                })

        return {
            "loading_plans": results,
            "trucks_used": trucks_used,
            "trucks_saved": trucks_saved,
            "baseline_trucks": baseline_trucks,
            "avg_utilization_percent": round(avg_utilization, 2),
            "baseline_utilization_percent": 80.0,
            "cost_saved_per_day_inr": round(trucks_saved * 650, 2),
            "unassigned_orders": unassigned_details
        }
           