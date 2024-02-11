import heapq

def print_puzzle(board):
    for row in board:
        print(" ".join(map(str, row)))
    print()

def manhattan_distance(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_position = next((row, col) for row, row_values in enumerate(goal) for col, col_value in enumerate(row_values) if col_value == value)
                distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
    return distance

def is_goal(state, goal):
    return state == goal

def get_neighbors(state):
    neighbors = []
    i, j = next((i, j) for i, row in enumerate(state) for j, value in enumerate(row) if value == 0)
    
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            new_state = [row.copy() for row in state]
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            neighbors.append(new_state)
    
    return neighbors

def astar(initial_state, goal_state):
    priority_queue = [(manhattan_distance(initial_state, goal_state), 0, initial_state)]
    visited = set()

    while priority_queue:
        _, cost, current_state = heapq.heappop(priority_queue)

        print(f"Step {cost}:")
        print_puzzle(current_state)

        if is_goal(current_state, goal_state):
            return current_state

        visited.add(tuple(map(tuple, current_state)))

        for neighbor in get_neighbors(current_state):
            if tuple(map(tuple, neighbor)) not in visited:
                heapq.heappush(priority_queue, (cost + 1 + manhattan_distance(neighbor, goal_state), cost + 1, neighbor))

    return None

# Main program
initial_state = [
    [1, 2, 3],
    [5, 0, 6],
    [8, 4, 7]
]

goal_state= [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]
print("Initial State:")
print_puzzle(initial_state)

result = astar(initial_state, goal_state)

if result is not None:
    print("Goal State:")
    print_puzzle(result)
else:
    print("No solution found.")
