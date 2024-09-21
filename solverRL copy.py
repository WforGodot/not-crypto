import numpy as np


class SolverRL:
    def __init__(self, solver):
        self.solver = solver
    
    def encode_state(self):
        state = []

        # Encode legend
        legend = []
        for item in self.solver.legend:
            for char in item:
                legend.append(ord(char)/255.0)
        state.extend(legend)

        # Encode constraints
        for constraint in self.solver.cons:
            state.extend(constraint)

        # Encode possibilities
        for poss in self.solver.poss:
            poss_vec = [0] * 10
            for val in poss:
                poss_vec[val] = 1
            state.extend(poss_vec)

        # Encode solved values
        if self.solver.solved is None:
            solved = [-1] * len(self.solver.legend)
        else:
            solved = [val if val is not None else -1 for val in self.solver.solved]
        state.extend(solved)

        # Encode assumptions
        assumptions = [item for sublist in self.solver.assumptions for item in sublist]
        state.extend(assumptions)

        # Encode carry
        state.extend([val if val is not None else -1 for val in self.solver.carry])

        # Ensure no NaN values in the state
        state = np.nan_to_num(state)

        return np.array(state, dtype=np.float32)


    def decode_action(self, action):
        action_type = action[0]

        if action_type == 0:  # find_poss
            return ("find_poss", action[1], action[2])
        elif action_type == 1:  # unify_cons
            return ("unify_cons", action[1], action[2], action[3])
        elif action_type == 2:  # make_assumption
            return ("make_assumption", action[1], action[2])
        elif action_type == 3:  # backtrack
            return ("backtrack",)

    def encode_action(self, action):
        if action[0] == "find_poss":
            return [0, action[1], action[2], 0]  # zero-padded
        elif action[0] == "unify_cons":
            return [1, action[1], action[2], action[3]]
        elif action[0] == "make_assumption":
            return [2, action[1], action[2], 0]  # zero-padded
        elif action[0] == "backtrack":
            return [3, 0, 0, 0]  # zero-padded