import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import networkx as nx

@dataclass
class Point:
    id: str

class TSPJsonSolver:
    def __init__(self):
        self.points = []
        self.graph = nx.Graph()
        
    def load_points(self, json_input: str) -> None:
        """Load points and distances from JSON string"""
        data = json.loads(json_input)
        self.points = [Point(p) for p in data['points']]
        
        # Build graph directly from distances
        for connection in data['distances']:
            point1 = connection['from']
            point2 = connection['to']
            distance = connection['distance']
            self.graph.add_edge(point1, point2, weight=distance)
    
    def build_graph(self) -> None:
        """No need to build graph as it's built in load_points"""
        pass
                    
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
        
        # Create response
        response = {
            "tour": tour,
            "total_distance": float(total_distance),
            "points": [{"id": p.id} for p in self.points]
        }
        
        return response

def main():
    # Example input JSON
    input_json = '''
    {
        "points": ["A", "B", "C", "D"],
        "distances": [
            {"from": "A", "to": "B", "distance": 1.4142},
            {"from": "B", "to": "C", "distance": 1.4142},
            {"from": "C", "to": "D", "distance": 2.0},
            {"from": "D", "to": "A", "distance": 2.0},
            {"from": "A", "to": "C", "distance": 2.8284},
            {"from": "B", "to": "D", "distance": 2.2361}
        ]
    }
    '''
    
    # Create solver
    solver = TSPJsonSolver()
    
    # Load points and solve
    solver.load_points(input_json)
    result = solver.solve_tsp()
    
    # Output result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()