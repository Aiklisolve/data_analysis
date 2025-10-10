# src/route_optimizer_v2.py
"""
RouteOptimizerV2: Multi-Start K-Means + NN + 2-Opt + Inter-Route + Load Balancing
FIXED: Handles 1 customer, prevents duplicate merged_customers, optimal k selection
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Tuple, Set
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class RouteOptimizerV2:
    """Route optimizer with multi-start k-means for optimal truck selection."""
    
    def __init__(
        self, 
        road_correction_factor: float = 1.3,
        fuel_cost_per_liter: float = 100.0,
        fuel_efficiency_km_per_liter: float = 8.0,
        avg_speed_kmph: float = 40.0,
        max_customers_per_route: int = 15,
        max_available_trucks: int = 15
    ):
        self.ROAD_CORRECTION_FACTOR = float(road_correction_factor)
        self.fuel_cost_per_liter = float(fuel_cost_per_liter)
        self.fuel_efficiency = float(fuel_efficiency_km_per_liter)
        self.avg_speed_kmph = float(avg_speed_kmph)
        self.max_customers_per_route = int(max_customers_per_route)
        self.max_available_trucks = int(max_available_trucks)


    @staticmethod
    def _deg2rad(deg: float) -> float:
        """Convert degrees to radians."""
        return deg * (math.pi / 180.0)


    def haversine_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float
    ) -> float:
        """Calculate great-circle distance with road correction."""
        R = 6371.0
        phi1 = self._deg2rad(lat1)
        phi2 = self._deg2rad(lat2)
        dphi = self._deg2rad(lat2 - lat1)
        dlambda = self._deg2rad(lon2 - lon1)
        
        a = (math.sin(dphi / 2.0) ** 2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
        c = 2.0 * math.asin(math.sqrt(a))
        straight_km = R * c
        
        return straight_km * self.ROAD_CORRECTION_FACTOR


    def create_distance_matrix(self, locations: List[Dict]) -> np.ndarray:
        """Create symmetric distance matrix for all locations."""
        n = len(locations)
        mat = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self.haversine_distance(
                    float(locations[i]['latitude']),
                    float(locations[i]['longitude']),
                    float(locations[j]['latitude']),
                    float(locations[j]['longitude'])
                )
                mat[i, j] = d
                mat[j, i] = d
                
        return mat


    def merge_colocated_customers(self, locations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Merge customers at identical coordinates.
        FIXED: Prevents duplicate customer_ids in merged_customers list.
        """
        depot = locations[0]
        customers = locations[1:]
        
        location_map = {}
        merged = [depot]
        customer_mapping = {}
        processed_customers = set()  # FIX: Track which customers we've processed
        
        for loc in customers:
            customer_id = loc.get('customer_id')
            
            # FIX: Skip if we already processed this customer_id
            if customer_id in processed_customers:
                continue
            
            processed_customers.add(customer_id)
            
            key = (round(float(loc['latitude']), 4), round(float(loc['longitude']), 4))
            
            if key in location_map:
                existing_idx = location_map[key]
                if 'merged_customers' not in merged[existing_idx]:
                    merged[existing_idx]['merged_customers'] = [merged[existing_idx]['customer_id']]
                
                # FIX: Only add if not already in list (extra safety)
                if customer_id not in merged[existing_idx]['merged_customers']:
                    merged[existing_idx]['merged_customers'].append(customer_id)
                
                customer_mapping[customer_id] = existing_idx
            else:
                loc['merged_customers'] = [customer_id]
                merged.append(loc)
                location_map[key] = len(merged) - 1
                customer_mapping[customer_id] = len(merged) - 1
        
        print(f"[debug] Merged {len(customers)} orders into {len(merged)-1} unique locations")
        
        # Print merged location details
        for loc in merged[1:]:
            if len(loc.get('merged_customers', [])) > 1:
                print(f"[debug]   {loc['customer_id']}: merged {len(loc['merged_customers'])} customers: {loc['merged_customers']}")
        
        return merged, customer_mapping


    def nearest_neighbor_route(
        self, 
        distance_matrix: np.ndarray, 
        start: int = 0
    ) -> List[int]:
        """Construct route using nearest neighbor heuristic."""
        n = distance_matrix.shape[0]
        if n == 0:
            return []
            
        unvisited: Set[int] = set(range(n))
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
        """Calculate total distance for a route."""
        return sum(mat[route[i], route[i + 1]] for i in range(len(route) - 1))


    def two_opt_with_jump_penalty(
        self, 
        route: List[int], 
        mat: np.ndarray, 
        max_iter: int = 200,
        max_jump_km: float = 12.0,
        jump_penalty_multiplier: float = 1.5
    ) -> List[int]:
        """2-opt with penalty for long jumps."""
        def penalized_route_distance(route_indices):
            total = 0.0
            for i in range(len(route_indices) - 1):
                dist = mat[route_indices[i], route_indices[i + 1]]
                if dist > max_jump_km:
                    dist *= jump_penalty_multiplier
                total += dist
            return total
        
        best = list(route)
        best_distance = penalized_route_distance(best)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    if j - i == 1:
                        continue
                        
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    new_dist = penalized_route_distance(new_route)
                    
                    if new_dist < best_distance - 1e-9:
                        best = new_route
                        best_distance = new_dist
                        improved = True
                        break
                        
                if improved:
                    break
                    
        return best


    def optimize_single_route(
        self, 
        locations: List[Dict]
    ) -> Tuple[List[Dict], float, float, float]:
        """Optimize a single route using NN + 2-Opt with jump penalty."""
        if len(locations) <= 1:
            return [], 0.0, 0.0, 0.0

        mat = self.create_distance_matrix(locations)
        initial = self.nearest_neighbor_route(mat, start=0)
        improved = self.two_opt_with_jump_penalty(initial, mat)
        total_distance = self._route_distance(improved, mat)
        
        fuel_liters = total_distance / max(1e-9, self.fuel_efficiency)
        fuel_cost = fuel_liters * self.fuel_cost_per_liter

        stops = []
        for idx in improved[1:-1]:
            stop_data = {
                "customer_id": locations[idx].get("customer_id"),
                "customer_name": locations[idx].get("customer_name"),
                "latitude": float(locations[idx]["latitude"]),
                "longitude": float(locations[idx]["longitude"])
            }
            if 'merged_customers' in locations[idx]:
                stop_data['merged_customers'] = locations[idx]['merged_customers']
            stops.append(stop_data)

        return stops, round(total_distance, 3), round(fuel_liters, 3), round(fuel_cost, 2)


    def _estimate_solution_distance(
        self, 
        locations: List[Dict], 
        clusters: List[List[int]]
    ) -> float:
        """Quick estimate of total distance for a clustering solution."""
        depot = locations[0]
        depot_lat = float(depot['latitude'])
        depot_lon = float(depot['longitude'])
        
        total_distance = 0.0
        
        for cluster in clusters:
            if not cluster:
                continue
            
            cluster_lats = [float(locations[idx]['latitude']) for idx in cluster]
            cluster_lons = [float(locations[idx]['longitude']) for idx in cluster]
            centroid_lat = sum(cluster_lats) / len(cluster_lats)
            centroid_lon = sum(cluster_lons) / len(cluster_lons)
            
            depot_to_centroid = self.haversine_distance(
                depot_lat, depot_lon, centroid_lat, centroid_lon
            )
            
            intra_cluster_dist = 0.0
            for i, idx_i in enumerate(cluster):
                for idx_j in cluster[i+1:]:
                    intra_cluster_dist += self.haversine_distance(
                        float(locations[idx_i]['latitude']),
                        float(locations[idx_i]['longitude']),
                        float(locations[idx_j]['latitude']),
                        float(locations[idx_j]['longitude'])
                    )
            
            estimated_route_dist = (depot_to_centroid * 2) + (intra_cluster_dist / len(cluster) if cluster else 0)
            total_distance += estimated_route_dist
        
        return total_distance


    def _multi_start_kmeans_clustering(
        self, 
        locations: List[Dict], 
        depot_lat: float, 
        depot_lon: float
    ) -> List[List[int]]:
        """
        Multi-Start K-Means: Try different k values and pick the best.
        FIXED: Handles 1 customer case, allows k=1, prevents crashes.
        """
        customers = locations[1:]
        n_customers = len(customers)
        
        if n_customers == 0:
            print("[debug] No customers - returning empty routes")
            return []
        
        # FIXED: Special case for 1 customer
        if n_customers == 1:
            print("[debug] Only 1 customer - using 1 truck")
            return [[1]]
        
        coords = np.array([
            [float(c['latitude']), float(c['longitude'])] 
            for c in customers
        ])
        
        # FIXED: Smart k-range calculation
        suggested_min = max(1, math.ceil(n_customers / self.max_customers_per_route))
        min_k = suggested_min
        
        # Can't have more clusters than customers
        max_k = min(
            n_customers,
            self.max_available_trucks,
            max(min_k + 4, n_customers // 3)
        )
        
        # Ensure valid range
        if max_k < min_k:
            max_k = min_k
        
        print(f"[debug] Multi-Start K-Means: n_customers={n_customers}, testing k={min_k} to k={max_k}")
        
        best_k = None
        best_total_distance = float('inf')
        best_clusters = None
        
        # Try each k value
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                labels = kmeans.fit_predict(coords)
                
                clusters = [[] for _ in range(k)]
                for idx, label in enumerate(labels):
                    clusters[label].append(idx + 1)
                
                clusters = [c for c in clusters if c]
                
                estimated_distance = self._estimate_solution_distance(locations, clusters)
                
                print(f"[debug]   k={k}: {len(clusters)} clusters, estimated ~{estimated_distance:.1f} km")
                
                if estimated_distance < best_total_distance:
                    best_total_distance = estimated_distance
                    best_k = k
                    best_clusters = clusters
            
            except Exception as e:
                print(f"[debug]   k={k}: Failed - {e}")
                continue
        
        if best_clusters is None:
            print("[WARNING] K-Means failed for all k values, falling back to single cluster")
            return [[i for i in range(1, n_customers + 1)]]
        
        print(f"[debug] ✅ Selected k={best_k} with ~{best_total_distance:.1f} km")
        
        sizes = [len(c) for c in best_clusters]
        print(f"[debug] Final cluster sizes: {sizes}")
        
        return best_clusters


    def inter_route_optimization(
        self, 
        routes_data: List[Dict], 
        locations: List[Dict],
        distance_matrix: np.ndarray,
        max_iterations: int = 30
    ) -> List[Dict]:
        """Swap customers between routes for improvement."""
        print(f"[debug] Starting inter-route optimization...")
        
        improved = True
        iteration = 0
        total_improvements = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(routes_data)):
                for j in range(i + 1, len(routes_data)):
                    for idx_i in range(len(routes_data[i]['stops'])):
                        for idx_j in range(len(routes_data[j]['stops'])):
                            
                            current_total = (routes_data[i]['total_distance_km'] + 
                                           routes_data[j]['total_distance_km'])
                            
                            test_stops_i = routes_data[i]['stops'].copy()
                            test_stops_j = routes_data[j]['stops'].copy()
                            
                            test_stops_i[idx_i], test_stops_j[idx_j] = test_stops_j[idx_j], test_stops_i[idx_i]
                            
                            depot = locations[0]
                            test_route_i = [depot] + [s for s in test_stops_i]
                            test_route_j = [depot] + [s for s in test_stops_j]
                            
                            new_dist_i = self._calculate_route_distance_from_stops(test_route_i, distance_matrix, locations)
                            new_dist_j = self._calculate_route_distance_from_stops(test_route_j, distance_matrix, locations)
                            new_total = new_dist_i + new_dist_j
                            
                            if new_total < current_total - 0.1:
                                routes_data[i]['stops'] = test_stops_i
                                routes_data[j]['stops'] = test_stops_j
                                routes_data[i]['total_distance_km'] = round(new_dist_i, 3)
                                routes_data[j]['total_distance_km'] = round(new_dist_j, 3)
                                
                                routes_data[i]['fuel_liters'] = round(new_dist_i / self.fuel_efficiency, 3)
                                routes_data[i]['fuel_cost_inr'] = round(routes_data[i]['fuel_liters'] * self.fuel_cost_per_liter, 2)
                                routes_data[i]['estimated_time_hours'] = round(new_dist_i / self.avg_speed_kmph, 3)
                                routes_data[i]['num_stops'] = len(test_stops_i)
                                
                                routes_data[j]['fuel_liters'] = round(new_dist_j / self.fuel_efficiency, 3)
                                routes_data[j]['fuel_cost_inr'] = round(routes_data[j]['fuel_liters'] * self.fuel_cost_per_liter, 2)
                                routes_data[j]['estimated_time_hours'] = round(new_dist_j / self.avg_speed_kmph, 3)
                                routes_data[j]['num_stops'] = len(test_stops_j)
                                
                                improved = True
                                total_improvements += 1
                                print(f"[debug] Swap improved: {current_total:.2f} → {new_total:.2f} km")
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        print(f"[debug] Inter-route optimization: {total_improvements} improvements made")
        return routes_data


    def _calculate_route_distance_from_stops(self, route_with_depot: List[Dict], 
                                            distance_matrix: np.ndarray, 
                                            all_locations: List[Dict]) -> float:
        """Calculate route distance from stop dictionaries."""
        total_dist = 0.0
        for i in range(len(route_with_depot) - 1):
            lat1 = float(route_with_depot[i]['latitude'])
            lon1 = float(route_with_depot[i]['longitude'])
            lat2 = float(route_with_depot[i + 1]['latitude'])
            lon2 = float(route_with_depot[i + 1]['longitude'])
            total_dist += self.haversine_distance(lat1, lon1, lat2, lon2)
        
        lat1 = float(route_with_depot[-1]['latitude'])
        lon1 = float(route_with_depot[-1]['longitude'])
        lat2 = float(route_with_depot[0]['latitude'])
        lon2 = float(route_with_depot[0]['longitude'])
        total_dist += self.haversine_distance(lat1, lon1, lat2, lon2)
        
        return total_dist


    def balance_routes(self, routes_data: List[Dict], locations: List[Dict]) -> List[Dict]:
        """GENTLE load balancing - allows ±3 stops variance."""
        if len(routes_data) < 2:
            return routes_data
        
        total_stops = sum(len(r['stops']) for r in routes_data)
        target_stops = total_stops / len(routes_data)
        max_stop_variance = 3
        
        print(f"[debug] Gentle balancing: target={target_stops:.1f} stops/truck (±{max_stop_variance})")
        
        max_iterations = 15
        iterations = 0
        max_distance_increase_allowed = 2.0
        
        while iterations < max_iterations:
            iterations += 1
            balanced = True
            
            sorted_routes = sorted(enumerate(routes_data), 
                                  key=lambda x: len(x[1]['stops']), 
                                  reverse=True)
            
            heaviest_idx, heaviest = sorted_routes[0]
            lightest_idx, lightest = sorted_routes[-1]
            
            stop_diff = len(heaviest['stops']) - len(lightest['stops'])
            
            if stop_diff > max_stop_variance:
                best_move_idx = -1
                best_distance_increase = float('inf')
                
                for idx, stop in enumerate(heaviest['stops']):
                    test_stops_heavy = heaviest['stops'][:idx] + heaviest['stops'][idx+1:]
                    test_stops_light = lightest['stops'] + [stop]
                    
                    depot = locations[0]
                    
                    if test_stops_heavy:
                        new_dist_heavy = self._calculate_route_distance_from_stops(
                            [depot] + test_stops_heavy, None, locations
                        )
                    else:
                        new_dist_heavy = 0
                        
                    new_dist_light = self._calculate_route_distance_from_stops(
                        [depot] + test_stops_light, None, locations
                    )
                    
                    current_total = heaviest['total_distance_km'] + lightest['total_distance_km']
                    new_total = new_dist_heavy + new_dist_light
                    distance_increase = new_total - current_total
                    
                    if distance_increase < best_distance_increase:
                        best_distance_increase = distance_increase
                        best_move_idx = idx
                
                if best_move_idx >= 0 and best_distance_increase <= max_distance_increase_allowed:
                    customer = heaviest['stops'].pop(best_move_idx)
                    lightest['stops'].append(customer)
                    
                    depot = locations[0]
                    
                    heaviest['total_distance_km'] = round(
                        self._calculate_route_distance_from_stops([depot] + heaviest['stops'], None, locations), 3
                    )
                    lightest['total_distance_km'] = round(
                        self._calculate_route_distance_from_stops([depot] + lightest['stops'], None, locations), 3
                    )
                    
                    heaviest['num_stops'] = len(heaviest['stops'])
                    lightest['num_stops'] = len(lightest['stops'])
                    
                    print(f"[debug] Balanced: Truck {heaviest_idx+1} to Truck {lightest_idx+1}, +{best_distance_increase:.2f} km")
                    balanced = False
                else:
                    print(f"[debug] Skipped: impact too high ({best_distance_increase:.2f} km)")
                    break
            
            if balanced:
                break
        
        stop_counts = [len(r['stops']) for r in routes_data]
        distances = [r['total_distance_km'] for r in routes_data]
        print(f"[debug] Final: stops={stop_counts}, distances={[f'{d:.1f}' for d in distances]} km")
        
        return routes_data


    def optimize_routes(self, locations: List[Dict]) -> Dict:
        """Main optimization with multi-start k-means."""
        if len(locations) <= 1:
            return {"routes": [], "total_distance_km": 0.0, "num_trucks_used": 0}

        depot = locations[0]
        merged_locations, customer_mapping = self.merge_colocated_customers(locations)
        customers = merged_locations[1:]
        
        if not customers:
            return {"routes": [], "total_distance_km": 0.0, "num_trucks_used": 0}

        depot_lat = float(depot['latitude'])
        depot_lon = float(depot['longitude'])
        
        # Use Multi-Start K-Means
        clusters = self._multi_start_kmeans_clustering(merged_locations, depot_lat, depot_lon)
        
        print(f"[debug] Multi-Start K-Means created {len(clusters)} optimal clusters")
        
        total_distance = 0.0
        routes_output: List[Dict] = []
        
        for truck_idx, cluster_indices in enumerate(clusters):
            seen = set()
            unique_indices = []
            for idx in cluster_indices:
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
            
            subset = [depot] + [merged_locations[idx] for idx in unique_indices]
            stops, dist, fuel_liters, fuel_cost = self.optimize_single_route(subset)
            total_distance += dist
            est_time_hours = dist / max(1e-9, self.avg_speed_kmph)
            
            routes_output.append({
                "truck_id": f"Truck_{truck_idx + 1}",
                "stops": stops,
                "total_distance_km": dist,
                "estimated_time_hours": round(est_time_hours, 3),
                "fuel_liters": fuel_liters,
                "fuel_cost_inr": fuel_cost,
                "num_stops": len(stops)
            })
        
        distance_matrix = self.create_distance_matrix(merged_locations)
        routes_output = self.inter_route_optimization(routes_output, merged_locations, distance_matrix)
        routes_output = self.balance_routes(routes_output, merged_locations)
        
        total_distance = sum(r['total_distance_km'] for r in routes_output)
        
        return {
            "routes": routes_output,
            "total_distance_km": round(total_distance, 3),
            "num_trucks_used": len(routes_output)
        }
