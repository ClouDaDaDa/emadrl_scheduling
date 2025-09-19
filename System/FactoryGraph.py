import math
import numpy as np
from local_realtime_scheduling.Environment.path_planning import a_star_search
import random


class Node:
    def __init__(self, x, y, walkable=True):
        self.x = x
        self.y = y
        self.walkable = walkable  # True if transbots can pass through
        self.g = float('inf')  # Cost from start node
        self.h = 0  # Heuristic cost to goal
        self.f = float('inf')  # Total cost
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f  # Priority in the open list


class FactoryGraph:
    def __init__(self, n_machines):
        nearest_square_root = math.ceil(n_machines ** 0.5)
        self.width = 6 * nearest_square_root + 2

        self.obstacles = []
        self.pickup_dropoff_points = {}
        self.location_index_map = self.create_location_to_index_map(n_machines)
        for machine_k in range(n_machines):
            x_index = machine_k % nearest_square_root
            y_index = machine_k // nearest_square_root
            for j in range(3):
                self.obstacles.append((2 + 6 * x_index + j, 3 + 3 * y_index))
            self.pickup_dropoff_points[f"machine_{machine_k}"] = (5 + 6 * x_index, 3 + 3 * y_index)
 
        charging_station_x_index = int(self.width / 2.0)
        # charging_station_0
        self.obstacles.append((charging_station_x_index, 0))
        self.pickup_dropoff_points["charging_0"] = (charging_station_x_index + 1, 0)
        # charging_station_1
        charging_station_1_y_index = 6 + 3 * ((n_machines - 1) // nearest_square_root)
        self.obstacles.append((charging_station_x_index, charging_station_1_y_index))
        self.pickup_dropoff_points["charging_1"] = (charging_station_x_index + 1, charging_station_1_y_index)

        self.pickup_dropoff_points["warehouse"] = (0, 0)

        self.pd_points_index = {value: key for key, value in self.pickup_dropoff_points.items()}

        self.height = charging_station_1_y_index + 1

        # print(f"pickup_dropoff_points: {self.pickup_dropoff_points}")

        # Initialize nodes as walkable by default
        self.nodes = [[Node(x, y) for y in range(self.height)] for x in range(self.width)]

        for obstacle_location in self.obstacles:
            self.set_obstacle(obstacle_location)

        self.unload_transport_time_matrix = self.compute_path_length_matrix()
        self.loaded_transport_time_matrix = self.unload_transport_time_matrix * 1.0

    def set_obstacle(self, location):
        x, y = location
        self.nodes[x][y].walkable = False  # Set as obstacle

    def set_walkable(self, location):
        x, y = location
        self.nodes[x][y].walkable = True  # Set as walkable

    def is_walkable(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.nodes[x][y].walkable

    def neighbors(self, node):
        # Return adjacent nodes that are walkable
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        result = []
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            if self.is_walkable(nx, ny):
                result.append(self.nodes[nx][ny])
        return result

    def reset_nodes(self):
        # Reset costs and parents for a new pathfinding run
        for row in self.nodes:
            for node in row:
                node.g = float('inf')
                node.h = 0
                node.f = float('inf')
                node.parent = None

    def compute_manhattan_distance_matrix(self):
        """
        Compute the Manhattan distance between all pickup/dropoff points.
        Returns a symmetric matrix where element [i][j] is the Manhattan distance
        between point i and point j.
        """
        points = list(self.pickup_dropoff_points.values())
        n_points = len(points)

        # Initialize a symmetric matrix of size n_points x n_points with zeros
        distance_matrix = np.zeros((n_points, n_points), dtype=int)

        # Compute distances
        for i in range(n_points):
            for j in range(i + 1, n_points):  # Compute upper triangular only
                distance = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance  # Symmetric matrix

        return distance_matrix

    def compute_path_length_matrix(self):
        """
        Compute the path length between all pickup/dropoff points using A*.
        Returns a symmetric matrix where element [i][j] is the path length
        between point i and point j.
        """
        points = list(self.pickup_dropoff_points.values())
        n_points = len(points)

        # Initialize a symmetric matrix of size n_points x n_points with zeros
        path_length_matrix = np.zeros((n_points, n_points), dtype=int)

        # Compute path lengths
        for i in range(n_points):
            for j in range(i + 1, n_points):
                path = a_star_search(self, points[i], points[j])
                # path_length = len(path) - 1 if path else float('inf')
                path_length = len(path) if path else float('inf')
                path_length_matrix[i][j] = path_length
                path_length_matrix[j][i] = path_length  # Symmetric matrix

        return path_length_matrix

    def create_location_to_index_map(self, n_machines):
        location_to_index = {}

        # First map all machines (they are the first in order)
        for i in range(n_machines):
            location_to_index[f'machine_{i}'] = i

        # Then map the charging stations (they are after the machines)
        n_charging = sum(1 for k in self.pickup_dropoff_points if k.startswith('charging_'))
        for i in range(n_charging):
            location_to_index[f'charging_{i}'] = n_machines + i

        # Finally map warehouse (it's at the end)
        location_to_index['warehouse'] = n_machines + n_charging

        return location_to_index

    def nearest_charging_station(self, location):
        x, y = location
        charging_0 = self.pickup_dropoff_points["charging_0"]
        charging_1 = self.pickup_dropoff_points["charging_1"]
        
        # Calculate the Euclidean distance to each charging station
        distance_to_charging_0 = abs(charging_0[0] - x) + abs(charging_0[1] - y) 
        distance_to_charging_1 = abs(charging_1[0] - x) + abs(charging_1[1] - y)
        
        # Return the nearest charging station
        if distance_to_charging_0 < distance_to_charging_1:
            return "charging_0"
        elif distance_to_charging_0 > distance_to_charging_1:
            return "charging_1"
        else:
            return random.choice(["charging_0", "charging_1"])

    def check_adjacent_positions_walkable(self, current_location, occupied_location):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # The direction of the given position relative to the current position
        given_direction = (occupied_location[0] - current_location[0],
                           occupied_location[1] - current_location[1])

        # todo: check whether the given_direction is valid
        if given_direction not in directions:
            raise ValueError(f"given_direction {given_direction} is invalid!")
        # Starting from given_direction, check the other three directions clockwise
        start_idx = directions.index(given_direction)

        for i in range(1, 4):
            # Calculate the index of the next direction to check (going clockwise)
            direction_idx = (start_idx + i) % 4
            dx, dy = directions[direction_idx]

            # Calculate the coordinates of the new position
            new_x = current_location[0] + dx
            new_y = current_location[1] + dy

            # If the location is walkable, return immediately
            if self.is_walkable(new_x, new_y) and ((new_x, new_y) not in self.pickup_dropoff_points.values()):
                return (dx, dy)

        # If no walkable location is found
        return None

if __name__ == "__main__":
    # Example usage of FactoryGraph
    n_machines = 5
    factory_graph = FactoryGraph(n_machines)

    # Print the pickup and dropoff points
    print("Pickup and Dropoff Points:")
    for name, point in factory_graph.pickup_dropoff_points.items():
        print(f"{name}: {point}")

    # Compute and print the Manhattan distance matrix
    manhattan_distance_matrix = factory_graph.compute_manhattan_distance_matrix()
    print("\nManhattan Distance Matrix:")
    print(manhattan_distance_matrix)

    # Compute and print the path length matrix
    path_length_matrix = factory_graph.compute_path_length_matrix()
    print("\nPath Length Matrix:")
    print(path_length_matrix)

    # Find and print the path between two specific points
    start_point = factory_graph.pickup_dropoff_points["machine_0"]
    goal_point = factory_graph.pickup_dropoff_points["warehouse"]
    path = a_star_search(factory_graph, start_point, goal_point)
    print(f"\nPath from {start_point} to {goal_point}:")
    print(path)
