import numpy as np
from queue import PriorityQueue

class PuzzleState:
    def __init__(self, current_value, parent_state):
        self.current_value = current_value
        self.parent_state = parent_state

    def __lt__(self, other):
        return False  # Define a default comparison method

class Puzzle:
    def __init__(self, start_value, goal_value):
        self.start_value = start_value
        self.goal_value = goal_value

    def print_state(self, state):
        print(state[:, :])

    def is_goal(self, state):
        return np.array_equal(state, self.goal_value)

    def get_possible_moves(self, state):
        possible_moves = []
        zero_pos = np.where(state == 0)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
        for direction in directions:
            new_pos = (zero_pos[0] + direction[0], zero_pos[1] + direction[1])
            if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:  # Check boundaries
                new_state = np.copy(state)
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]  # Swap
                possible_moves.append(new_state)
        return possible_moves

    def heuristic(self, state):
         return np.count_nonzero(state != self.goal_value)

    def solve(self):
        queue = PriorityQueue()
        start_state = PuzzleState(self.start_value, None)
        queue.put((0, start_state))  # Put State object in queue
        visited = set()

        while not queue.empty():
            priority, current_state = queue.get()
            if self.is_goal(current_state.current_value):
                return current_state  # Return final state
            for move in self.get_possible_moves(current_state.current_value):
                move_state = PuzzleState(move, current_state)  # Create new State for move
                if str(move_state.current_value) not in visited:
                    visited.add(str(move_state.current_value))
                    priority = self.heuristic(move_state.current_value)
                    queue.put((priority, move_state))  # Put State object in queue
        return None

# Test the function
print("Start matrix:")
start_value = np.array([[int(x) for x in input().split()] for _ in range(3)])
print("Goal matrix:")
goal_value = np.array([[int(x) for x in input().split()] for _ in range(3)])
puzzle = Puzzle(start_value, goal_value)
solution_state = puzzle.solve()
if solution_state is not None:
    moves = []
    while solution_state is not None:  # Go through parents to get moves
        moves.append(solution_state.current_value)
        solution_state = solution_state.parent_state
    for move in reversed(moves):  # Print moves in correct order
        puzzle.print_state(move)
else:
    print("No solution found.")
