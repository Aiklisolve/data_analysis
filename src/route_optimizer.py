from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Tuple
from copy import deepcopy

class RouteOptimizer:
    def __init__(self, road_correction_factor: float = 1.3,
                 fuel_cost_per_liter: float = 100.0,
                 fuel_efficiency_km_per_liter: float = 8.0,
                 avg_speed_kmph: float = 40.0):
        self.ROAD_CORRECTION_FACTOR = float(road_correction_factor)
        self.fuel_cost_per_liter = float(fuel_cost_per_liter)
        self.fuel_efficiency = float(fuel_efficiency_km_per_liter)
        self.avg_speed_kmph = float(avg_speed_kmph)

    @staticmethod
    def _deg2rad(deg: float) -> float:
        return deg * (math.pi / 180.0)

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        phi1 = self._deg2rad(lat1)
        phi2 = self._deg2rad(lat2)
        dphi = self._deg2rad(lat2 - lat1)
        dlambda = self._deg2rad(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
        c = 2.0 * math.asin(math.sqrt(a))
        straight_km = R * c
        return straight_km * self.ROAD_CORRECTION_FACTOR

    def create_distance_matrix(self, locations: List[Dict]) -> np.ndarray:
        n = len(locations)
        mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i+1, n):
                d = self.haversine_distance(
                    float(locations[i]['latitude']),
                    float(locations[i]['longitude']),
                    float(locations[j]['latitude']),
                    float(locations[j]['longitude'])
                )
                mat[i, j] = d
                mat[j, i] = d
        return mat

    def nearest_neighbor_route(self, distance_matrix: np.ndarray, start: int = 0) -> List[int]:
        n = distance_matrix.shape[0]
        if n == 0:
            return []
        unvisited = set(range(n))
        unvisited.remove(start)
        route = [start]
        current = start
        while unvisited:
            next_point = min(unvisited, key=lambda j: distance_matrix[current, j])
            route.append(next_point)
            unvisited.remove(next_point)
            current = next_point
        route.append(start)
        return route

    def _route_distance(self, route: List[int], mat: np.ndarray) -> float:
        return sum(mat[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(self, route: List[int], mat: np.ndarray, max_iter: int = 200) -> List[int]:
        best = list(route)
        best_distance = self._route_distance(best, mat)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_dist = self._route_distance(new_route, mat)
                    if new_dist < best_distance - 1e-9:
                        best = new_route
                        best_distance = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best

    def optimize_single_route(self, locations: List[Dict]) -> Tuple[List[Dict], float, float, float]:
        if len(locations) <= 1:
            return [], 0.0, 0.0, 0.0

        mat = self.create_distance_matrix(locations)
        initial = self.nearest_neighbor_route(mat, start=0)
        improved = self.two_opt(initial, mat)
        total_distance = self._route_distance(improved, mat)
        fuel_liters = total_distance / max(1e-9, self.fuel_efficiency)
        fuel_cost = fuel_liters * self.fuel_cost_per_liter
        est_time_hours = total_distance / max(1e-9, self.avg_speed_kmph)

        stops = []
        for idx in improved[1:-1]:
            stops.append({
                "customer_id": locations[idx].get("customer_id"),
                "customer_name": locations[idx].get("customer_name"),
                "latitude": float(locations[idx]["latitude"]),
                "longitude": float(locations[idx]["longitude"])
            })

        return stops, round(total_distance, 3), round(fuel_liters, 3), round(fuel_cost, 2)

    def optimize_multi_truck(self, locations: List[Dict], num_trucks: int) -> Dict:
        if len(locations) <= 1:
            return {"routes": [], "total_distance_km": 0.0}

        depot = locations[0]
        customers = locations[1:]
        n = len(customers)
        chunk_size = max(1, math.ceil(n / num_trucks))
        total_distance = 0.0
        routes_output: List[Dict] = []

        for i in range(num_trucks):
            sub_customers = customers[i*chunk_size:(i+1)*chunk_size]
            if not sub_customers:
                continue
            subset = [depot] + sub_customers
            stops, dist, fuel_liters, fuel_cost = self.optimize_single_route(subset)
            total_distance += dist
            est_time_hours = dist / max(1e-9, self.avg_speed_kmph)
            routes_output.append({
                "truck_id": f"Truck_{i+1}",
                "stops": stops,
                "total_distance_km": dist,
                "estimated_time_hours": round(est_time_hours, 3),
                "fuel_liters": fuel_liters,
                "fuel_cost_inr": fuel_cost
            })

        return {
            "routes": routes_output,
            "total_distance_km": round(total_distance, 3)
        }