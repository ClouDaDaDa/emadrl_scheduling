import heapq
import matplotlib.pyplot as plt


# def heuristic(node, goal):
#     # Use Manhattan distance as heuristic for A* on a grid
#     return abs(node.x - goal.x) + abs(node.y - goal.y)
#
#
# def a_star_search(graph, start, goal):
#     # Reset graph nodes for a fresh search
#     graph.reset_nodes()
#
#     # Initialize start node
#     start_node = graph.nodes[start[0]][start[1]]
#     goal_node = graph.nodes[goal[0]][goal[1]]
#     start_node.g = 0
#     start_node.h = heuristic(start_node, goal_node)
#     start_node.f = start_node.h
#
#     open_list = []
#     heapq.heappush(open_list, (start_node.f, start_node))
#
#     closed_set = set()
#
#     while open_list:
#         _, current = heapq.heappop(open_list)
#
#         # If the goal is reached, reconstruct path
#         if current == goal_node:
#             return reconstruct_path(goal_node)
#
#         closed_set.add((current.x, current.y))
#
#         for neighbor in graph.neighbors(current):
#             if (neighbor.x, neighbor.y) in closed_set:
#                 continue
#
#             tentative_g = current.g + 1  # Assume uniform cost of moving to a neighbor
#
#             if tentative_g < neighbor.g:
#                 neighbor.g = tentative_g
#                 neighbor.h = heuristic(neighbor, goal_node)
#                 neighbor.f = neighbor.g + neighbor.h
#                 neighbor.parent = current
#
#                 # If neighbor is not in open list, add it
#                 if not any(neighbor == item[1] for item in open_list):
#                     heapq.heappush(open_list, (neighbor.f, neighbor))
#
#     return []  # Return empty path if no path is found
#
#
# def reconstruct_path(goal_node):
#     # Reconstruct path from goal to start by following parent pointers
#     path = []
#     current = goal_node
#     while current:
#         path.append((current.x, current.y))
#         current = current.parent
#     path.reverse()  # Reverse to get path from start to goal
#     return path
#
#
# # Visualization function (Optional for better clarity)
# def visualize_path(graph, path, obstacles, start, goal):
#     plt.figure(figsize=(8, 8))
#     plt.xlim(-1, graph.width)
#     plt.ylim(-1, graph.height)
#     plt.gca().invert_yaxis()
#     # plt.grid(True)
#     # Set grid with spacing of 1 unit
#     plt.xticks(range(0, graph.width, 1))
#     plt.yticks(range(0, graph.height, 1))
#     plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
#
#     # Plot obstacles
#     for (x, y) in obstacles:
#         plt.plot(x, y, 's', color="black", markersize=25)
#
#     # Plot pickup/dropoff points with lighter, semi-transparent circles
#     for (_, (x, y)) in pickup_dropoff_points.items():
#         plt.plot(x, y, 'o', color="lightblue", markersize=20, alpha=0.8)
#
#     # Plot path
#     for (x, y) in path:
#         plt.plot(x, y, 'o', color="blue", markersize=10)
#
#     # Mark start and goal
#     plt.plot(start[0], start[1], 'go', markersize=12, label="Start")
#     plt.plot(goal[0], goal[1], 'ro', markersize=12, label="Goal")
#
#     plt.legend()
#     plt.show()


def heuristic(node, goal):
    # Use Manhattan distance as heuristic for A* on a grid
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def reconstruct_path(goal_node):
    # Reconstruct path from goal to start by following parent pointers
    path = []
    current = goal_node
    while current:
        path.append((current.x, current.y))
        current = current.parent
    path.reverse()  # Reverse to get path from start to goal
    path.pop(0)
    return path


def a_star_search(graph, start, goal):
    # Reset graph nodes for a fresh search
    graph.reset_nodes()

    # Temporarily mark pickup/dropoff points as obstacles (except start and goal)
    reserved_nodes = set(graph.pickup_dropoff_points.values()) - {start, goal}
    for node_location in reserved_nodes:
        graph.set_obstacle(location=node_location)

    # Initialize start and goal nodes
    start_node = graph.nodes[start[0]][start[1]]
    goal_node = graph.nodes[goal[0]][goal[1]]
    start_node.g = 0
    start_node.h = heuristic(start_node, goal_node)
    start_node.f = start_node.h

    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))

    closed_set = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        # If the goal is reached, reconstruct path
        if current == goal_node:
            # Restore reserved nodes as walkable
            for x, y in reserved_nodes:
                graph.nodes[x][y].walkable = True
            return reconstruct_path(goal_node)

        closed_set.add((current.x, current.y))

        for neighbor in graph.neighbors(current):
            if (neighbor.x, neighbor.y) in closed_set:
                continue

            tentative_g = current.g + 1  # Assume uniform cost of moving to a neighbor

            if tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, goal_node)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current

                # If neighbor is not in open list, add it
                if not any(neighbor == item[1] for item in open_list):
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    # Restore reserved nodes as walkable if no path is found
    for node_location in reserved_nodes:
        graph.set_walkable(location=node_location)

    # raise Exception(f"Cannot find a walkable path from {start} to {goal}!")
    return []  # Return empty path if no path is found


def heuristic_with_turn_penalty(node, goal, parent_direction=None):
    """
    Calculate the heuristic with an additional penalty for turns.
    """
    # Basic Manhattan distance as heuristic
    manhattan_distance = abs(node.x - goal.x) + abs(node.y - goal.y)

    # Add a penalty for turning
    if parent_direction:
        dx = node.x - parent_direction[0]
        dy = node.y - parent_direction[1]

        # A turn occurs if dx/dy differs from the parent's movement
        if dx != parent_direction[2] or dy != parent_direction[3]:
            turn_penalty = 1000  # Adjust this value to control the penalty strength
        else:
            turn_penalty = 0
    else:
        turn_penalty = 0

    return manhattan_distance + turn_penalty


def a_star_search_with_turn_penalty(graph, start, goal):
    """
    A* algorithm with turn penalties to reduce the number of turns in the path.
    """
    graph.reset_nodes()

    # Temporarily mark pickup/dropoff points as obstacles (except start and goal)
    reserved_nodes = set(graph.pickup_dropoff_points.values()) - {start, goal}
    for node_location in reserved_nodes:
        graph.set_obstacle(location=node_location)

    start_node = graph.nodes[start[0]][start[1]]
    goal_node = graph.nodes[goal[0]][goal[1]]
    start_node.g = 0
    start_node.h = heuristic(start_node, goal_node)
    start_node.f = start_node.h

    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node, None))  # Add parent direction to tuple

    closed_set = set()

    while open_list:
        _, current, parent_direction = heapq.heappop(open_list)

        # If the goal is reached, reconstruct path
        if current == goal_node:
            for x, y in reserved_nodes:
                graph.nodes[x][y].walkable = True
            return reconstruct_path(goal_node)

        closed_set.add((current.x, current.y))

        for neighbor in graph.neighbors(current):
            if (neighbor.x, neighbor.y) in closed_set:
                continue

            # Determine the direction of the current step
            current_direction = (current.x, current.y, neighbor.x - current.x, neighbor.y - current.y)
            tentative_g = current.g + 1  # Basic cost is still 1 per step

            # Adjust the heuristic to penalize turning
            neighbor.h = heuristic_with_turn_penalty(neighbor, goal_node, current_direction)
            neighbor.f = tentative_g + neighbor.h

            if tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.parent = current

                if not any(neighbor == item[1] for item in open_list):
                    heapq.heappush(open_list, (neighbor.f, neighbor, current_direction))

    # for x, y in reserved_nodes:
    #     graph.nodes[x][y].walkable = True
    for node_location in reserved_nodes:
        graph.set_walkable(location=node_location)

    return []


# # Example Usage
# if __name__ == "__main__":
#
#     # Set up a 16x16 factory floor with machines, charging stations, and transbot pathfinding
#     width, height = 16, 16
#     graph = FactoryGraph(width, height)
#
#     # Define obstacles (machines and charging stations)
#     obstacles = [
#         (2, 5), (3, 5), (4, 5),  # Machine 1 occupies these nodes
#         (8, 5), (9, 5), (10, 5),  # Machine 2
#         (2, 8), (3, 8), (4, 8),  # Machine 3
#         (8, 8), (9, 8), (10, 8),  # Machine 4
#         (2, 11), (3, 11), (4, 11),  # Machine 5
#         (8, 11), (9, 11), (10, 11),  # Machine 6
#         (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),  # Warehouse
#         (10, 2),  # Charging station 1
#         (5, 14)  # Charging station 2
#     ]
#
#     # Mark obstacles on the graph
#     for (x, y) in obstacles:
#         graph.set_obstacle(x, y)
#
#     # Define pick-up and drop-off points (next to each machine/charging station)
#     pickup_dropoff_points = {
#         "warehouse": (2, 2),
#         "machine_1": (5, 5),
#         "machine_2": (11, 5),
#         "machine_3": (5, 8),
#         "machine_4": (11, 8),
#         "machine_5": (5, 11),
#         "machine_6": (11, 11),
#         "charging_1": (11, 2),
#         "charging_2": (6, 14)
#     }
#
#     # Transbot initial and goal positions
#     start = pickup_dropoff_points["charging_1"]  # Starting position (e.g., warehouse)
#     goal = pickup_dropoff_points["machine_3"]  # Target machine for pickup/drop-off
#
#     # Run A* search
#     path = a_star_search(graph, start, goal)
#
#     # Output the path
#     print("Path from start to goal:", path)
#
#     # Visualize the path
#     visualize_path(graph, path, obstacles, start, goal)


