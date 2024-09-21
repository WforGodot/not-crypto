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

        # Encode valid actions as a fixed-length binary vector
        self.valid_actions = self.get_valid_actions()
        valid_actions_encoded = [0] * len(self.valid_actions)
        for i in range(len(self.valid_actions)):
            valid_actions_encoded[i] = 1
        state.extend(valid_actions_encoded)

        # Ensure no NaN values in the state
        state = np.nan_to_num(state)

        return np.array(state, dtype=np.float32)
    
    def get_valid_actions(self):
        valid_actions = []

        # find_poss actions
        for i, cons in enumerate(self.solver.cons):
            for j in [x for x in range(len(self.solver.legend)) if cons[x] != 0]:
                valid_actions.append(("find_poss", i, j))

        # unify_cons actions
        for i in range(len(self.solver.cons)):
            for j in range(i+1, len(self.solver.cons)):
                for x in [k for k in range(len(self.solver.legend)) if self.solver.cons[i][k] != 0 and self.solver.cons[j][k] != 0]:
                    valid_actions.append(("unify_cons", i, j, x))

        # make_assumption actions
        for i in range(len(self.solver.legend)):
            if self.solver.solved[i] is None:
                for val in self.solver.poss[i]:
                    valid_actions.append(("make_assumption", i, val))

        # backtrack action
        valid_actions.append(("backtrack",))

        return valid_actions

    def decode_action(self, action_index):
        if action_index < 0 or action_index >= len(self.valid_actions):
            raise ValueError("Invalid action index")
        return self.valid_actions[action_index]

    def encode_action(self, action):
        if action[0] == "find_poss":
            return [0, action[1], action[2], 0]  # zero-padded
        elif action[0] == "unify_cons":
            return [1, action[1], action[2], action[3]]
        elif action[0] == "make_assumption":
            return [2, action[1], action[2], 0]  # zero-padded
        elif action[0] == "backtrack":
            return [3, 0, 0, 0]  # zero-padded
