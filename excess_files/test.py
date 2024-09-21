# test_solver.py
from solver import Solver_backtrack

def test_cryptarithmetic():
    # Define the puzzle
    puzzle = "EAR + CAR = RICE"

    # Create an instance of Solver_backtrack with the given puzzle
    solver = Solver_backtrack(puzzle)

    # Attempt to solve the puzzle
    solution = solver.solve_with_backtracking()

    # Print the solution or an indication that no solution was found
    if solution:
        # Convert solution indices to actual letters based on solver's legend
        answer = ''.join(str(solution[solver.tree[-1].legend.index(char)]) if char in solver.tree[-1].legend else char for char in puzzle)
        print(f"Solved Puzzle: {answer}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    test_cryptarithmetic()
