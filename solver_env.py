import gymnasium
from gymnasium.spaces import Box, Discrete
from solverRL import SolverRL
from solver import SolverNL
import numpy as np

class CryptarithmeticEnv(gymnasium.Env):
    def __init__(self, problem):
        self.problem = problem
        self.solver = SolverNL(problem)
        self.solver_rl = SolverRL(self.solver)

        # Define action space dimensions based on the number of valid actions
        self.action_space = Discrete(len(self.solver_rl.get_valid_actions()))

        # Define observation space shape and bounds
        self.observation_space = Box(low=0, high=255, shape=(len(self.solver_rl.encode_state()),), dtype=np.float32)

        self.max_steps = 30
        self.current_step = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.solver = SolverNL(self.problem)
        self.solver_rl = SolverRL(self.solver)
        self.current_step = 0  # Reset step counter
        state = self.solver_rl.encode_state()
        info = {}
        return state, info

    def step(self, action):
        self.current_step += 1
        action_decoded = self.solver_rl.decode_action(action)
        self.solver.parse_and_execute_list([action_decoded])

        state = self.solver_rl.encode_state()
        done = self.solver.is_solved() or self.current_step >= self.max_steps

        # Calculate reward based on the number of solved items
        num_solved = sum(1 for item in self.solver.solved if item is not None)
        reward = num_solved * 1  # Reward for each solved item

        if self.solver.is_solved():
            reward += 100  # Additional reward for fully solving the problem
        elif self.current_step >= self.max_steps:
            reward = 0  # No reward if the maximum number of steps is reached without solving

        info = {}

        return state, reward, done, False, info

    def render(self):
        print(self.solver.get_log_and_state())

    def close(self):
        pass
