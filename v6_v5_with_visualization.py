import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

@dataclass
class Point:
    x: float
    y: str
    id: str

class TSPVisualSolver:
    def __init__(self):
        self.points = []
        self.graph = nx.Graph()
        self.distances = None
        
    def load_points(self, json_input: str) -> None:
        """Load points from JSON string"""
        data = json.loads(json_input)
        self.points = [
            Point(p['x'], p['y'], p.get('id', str(i)))
            for i, p in enumerate(data['points'])
        ]
        
    def build_graph(self) -> None:
        """Build graph from points"""
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if i != j:
                    dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    self.graph.add_edge(p1.id, p2.id, weight=dist)
                    
    def visualize_solution(self, tour: List[str]) -> None:
        """Visualize the TSP solution"""
        plt.figure(figsize=(10, 10))
        
        # Plot all points
        points_array = np.array([[p.x, p.y] for p in self.points])
        plt.scatter(points_array[:, 0], points_array[:, 1], c='red', s=100)
        
        # Plot point IDs
        for p in self.points:
            plt.annotate(p.id, (p.x, p.y), xytext=(5, 5), textcoords='offset points')
        
        # Plot tour path
        for i in range(len(tour)):
            point1 = next(p for p in self.points if p.id == tour[i])
            point2 = next(p for p in self.points if p.id == tour[(i+1) % len(tour)])
            plt.plot([point1.x, point2.x], [point1.y, point2.y], 'b-', linewidth=2)
        
        plt.title('TSP Solution')
        plt.grid(True)
        plt.show()
                    
    def solve_tsp(self) -> Dict:
        """Solve TSP and return JSON response"""
        # Get TSP tour using networkx
        tour = nx.approximation.traveling_salesman_problem(
            self.graph, cycle=True
        )
        
        # Calculate total distance
        total_distance = sum(
            self.graph[tour[i]][tour[i+1]]['weight']
            for i in range(len(tour)-1)
        )
        
        # Visualize solution
        self.visualize_solution(tour)
        
        # Create response
        response = {
            "tour": tour,
            "total_distance": float(total_distance),
            "points": [
                {"id": p.id, "x": p.x, "y": p.y}
                for p in self.points
            ]
        }
        
        return response
    
    def calculate_distance_matrix(self):
        n = len(self.points)
        self.distances = np.zeros((n, n))
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if i != j:
                    self.distances[i,j] = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                else:
                    self.distances[i,j] = np.inf
    
    def solve_tsp_homotopy(self) -> Dict:
        """Solve TSP using convex hull with graph distances"""
        n = len(self.points)
        
        # Create distance matrix from graph
        dist_matrix = np.full((n, n), np.inf)
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if self.graph.has_edge(p1.id, p2.id):
                    dist_matrix[i,j] = self.graph[p1.id][p2.id]['weight']
        
        # Find convex hull using graph distances
        def find_graph_hull():
            hull = []
            # Start with first point
            current = 0
            hull.append(current)
            
            while len(hull) < n:
                # Find point that forms largest angle with current edge
                current = hull[-1]
                if len(hull) == 1:
                    # For first point, just find nearest
                    next_point = min(range(n), 
                                   key=lambda x: dist_matrix[current][x] if x != current else np.inf)
                else:
                    prev = hull[-2]
                    candidates = set(range(n)) - set(hull)
                    if not candidates:
                        break
                    
                    # Use distances to approximate angles
                    next_point = min(candidates,
                                   key=lambda x: (dist_matrix[current][x] + 
                                                dist_matrix[x][prev] - 
                                                dist_matrix[current][prev])
                                   if dist_matrix[current][x] != np.inf and 
                                      dist_matrix[x][prev] != np.inf
                                   else np.inf)
                hull.append(next_point)
            
            return hull
        
        # Get initial tour from graph-based convex hull
        hull_tour = find_graph_hull()
        
        # Complete tour using nearest neighbor for remaining points
        remaining = set(range(n)) - set(hull_tour)
        tour = hull_tour
        
        while remaining:
            best_insertion = None
            best_cost = np.inf
            
            # Find best insertion point
            for point in remaining:
                for i in range(len(tour)):
                    prev = tour[i]
                    next_point = tour[(i + 1) % len(tour)]
                    
                    if (dist_matrix[prev][point] != np.inf and 
                        dist_matrix[point][next_point] != np.inf):
                        cost = (dist_matrix[prev][point] + 
                               dist_matrix[point][next_point] - 
                               dist_matrix[prev][next_point])
                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (point, i + 1)
            
            if best_insertion:
                point, pos = best_insertion
                tour.insert(pos, point)
                remaining.remove(point)
            else:
                # If no valid insertion found, append to end
                tour.extend(list(remaining))
                break
        
        # Convert to ID-based tour
        tour_ids = [self.points[i].id for i in tour]
        
        # Calculate total distance
        total_distance = sum(
            self.graph[tour_ids[i]][tour_ids[(i+1)%len(tour_ids)]]['weight']
            for i in range(len(tour_ids))
            if self.graph.has_edge(tour_ids[i], tour_ids[(i+1)%len(tour_ids)])
        )
        
        # Visualize solution
        self.visualize_solution(tour_ids)
        
        return {
            "tour": tour_ids,
            "total_distance": float(total_distance),
            "points": [{"id": p.id, "x": p.x, "y": p.y} for p in self.points]
        }
    
    def _get_tour_from_matrix(self, matrix):
        """Convert permutation matrix to tour"""
        n = len(matrix)
        tour = [0]
        visited = {0}
        
        for _ in range(n-1):
            current = tour[-1]
            next_indices = np.where(matrix[current] == 1)[0]
            if len(next_indices) == 0:
                # If no next city found, pick the nearest unvisited city
                unvisited = list(set(range(n)) - visited)
                distances_to_unvisited = [self.distances[current][j] for j in unvisited]
                next_city = unvisited[np.argmin(distances_to_unvisited)]
            else:
                next_city = next_indices[0]
            tour.append(next_city)
            visited.add(next_city)
        
        return tour

def main():
    # Example input JSON
    input_json = '''
    {
        "points": [
            {"x": 0, "y": 0, "id": "A"},
            {"x": 1, "y": 1, "id": "B"},
            {"x": 2, "y": 2, "id": "C"},
            {"x": 0, "y": 2, "id": "D"}
        ]
    }
    '''
    
    # Create solver
    solver = TSPVisualSolver()
    
    # Load points and solve
    solver.load_points(input_json)
    solver.build_graph()
    result = solver.solve_tsp_homotopy()
    
    # Output result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
