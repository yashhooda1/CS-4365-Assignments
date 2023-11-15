#Name: 8-puzzle solver
#Author: Yash Hooda
#Date Created: 11/15/2023

#Import Statements
import heapq
from copy import deepcopy

#PuzzleNode class to 
class PuzzleNode:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def is_goal_state(node):
    goal_state = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]  # Define your goal state
    return node.state == goal_state

def get_blank_position(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def get_neighbors(node):
    i, j = get_blank_position(node.state)
    neighbors = []

    # Define possible moves (up, down, left, right)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for move in moves:
        ni, nj = i + move[0], j + move[1]

        if 0 <= ni < 3 and 0 <= nj < 3:
            new_state = deepcopy(node.state)
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]

            neighbors.append(PuzzleNode(new_state, node, move, node.cost + 1, heuristic(node.state)))

    return neighbors

def dfs(initial_state):
    initial_node = PuzzleNode(initial_state)
    stack = [initial_node]
    visited = set()

    while stack:
        current_node = stack.pop()

        if is_goal_state(current_node):
            return current_node

        if current_node not in visited:
            visited.add(current_node)

            stack.extend(get_neighbors(current_node))

    return None

def ids(initial_state):
    depth = 0

    while True:
        result = depth_limited_dfs(initial_state, depth)

        if result:
            return result

        depth += 1

        if depth > 10:
            return None

def depth_limited_dfs(initial_state, depth_limit):
    initial_node = PuzzleNode(initial_state)
    stack = [(initial_node, 0)]
    visited = set()

    while stack:
        current_node, current_depth = stack.pop()

        if current_depth > depth_limit:
            continue

        if is_goal_state(current_node):
            return current_node

        if current_node not in visited:
            visited.add(current_node)

            neighbors = get_neighbors(current_node)

            for neighbor in neighbors:
                stack.append((neighbor, current_depth + 1))

    return None

def heuristic(state):
    # Define your heuristics here
    # Heuristic 1: Number of misplaced tiles
    # Heuristic 2: Manhattan distance
    # You can implement both and compare their performance in the analysis

def astar(initial_state, heuristic_function):
    initial_node = PuzzleNode(initial_state, heuristic=heuristic_function(initial_state))
    heap = [initial_node]
    visited = set()

    while heap:
        current_node = heapq.heappop(heap)

        if is_goal_state(current_node):
            return current_node

        if current_node not in visited:
            visited.add(current_node)

            neighbors = get_neighbors(current_node)

            for neighbor in neighbors:
                neighbor.heuristic = heuristic_function(neighbor.state)
                heapq.heappush(heap, neighbor)

    return None

# Example usage
if __name__ == "__main__":
    import sys

    algorithm = sys.argv[1]
    input_state = [int(x) if x != '*' else 0 for x in sys.argv[2:]]

    if algorithm == 'dfs':
        result = dfs([input_state[i:i+3] for i in range(0, len(input_state), 3)])
    elif algorithm == 'ids':
        result = ids([input_state[i:i+3] for i in range(0, len(input_state), 3)])
    elif algorithm == 'astar1':
        result = astar([input_state[i:i+3] for i in range(0, len(input_state), 3)], heuristic_function=heuristic1)
    elif algorithm == 'astar2':
        result = astar([input_state[i:i+3] for i in range(0, len(input_state), 3)], heuristic_function=heuristic2)
    else:
        print("Invalid algorithm name. Please choose dfs, ids, astar1, or astar2.")
        sys.exit(1)

    if result:
        # Print the solution
        moves = []
        while result:
            moves.insert(0, result.state)
            result = result.parent
        for move in moves:
            for row in move:
                print(' '.join(map(str, row)))
            print("\n")
        print(f"Number of moves: {len(moves) - 1}")
        print(f"Number of states enqueued: {len(visited)}")
    else:
        print("Goal state not found before or at depth 10.")
