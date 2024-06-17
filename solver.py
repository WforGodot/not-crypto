from dataclasses import dataclass
from math import lcm, prod
import itertools
from tabulate import tabulate
from typing import Any
import copy


def pad(x, n):
    return [None for i in range(n - len(x))] + x


class Solver():

    def __init__(self, init_dict):
        self.legend = init_dict['legend']
        self.uniques = init_dict['uniques']
        self.original_cons = init_dict['cons']
        self.cons = init_dict['cons']
        self.hidden_cons = []
        self.poss = init_dict['poss']
        self.solved = init_dict['solved']
        self.tried = init_dict['tried']
        self.error = ''
        self.valid = True
        self.assumptions = []
        self.carry = init_dict['carry']
        self.str1 = init_dict['str1']
        self.str2 = init_dict['str2']
        self.str3 = init_dict['str3']
        self.log = ''
    
    @classmethod
    def from_crypt(cls, string):
        split = string.split(" ")
        str_1, str_2, str_3 = split[0], split[2], split[4]
        carry = ["C"+str(i) for i in range(1, len(str_3))]
        legend_base = list(dict.fromkeys(str_1 + str_2 + str_3))
        uniques = len(legend_base)
        legend = legend_base + carry
        str1 = pad([legend.index(i) for i in str_1], len(str_3))
        str2 = pad([legend.index(i) for i in str_2], len(str_3))
        str3 = [legend.index(i) for i in str_3]
        carry = [legend.index(i) for i in carry] + [None]
        cons = []
        print(str1, str2, str3, carry, legend)
        for i in range(len(str3)):
            x = [0 for i in range(len(legend)+ 1)]
            if str1[i] != None:
                x[str1[i]] += 1
            if str2[i] != None:
                x[str2[i]] += 1
            if str3[i] != None:
                x[str3[i]] -= 1
            if carry[i] != None:
                x[carry[i]] += 1
            if i > 0:
                x[carry[i-1]] -= 10
            cons.append(x)
        
        heads = list(dict.fromkeys([legend.index(i) for i in [str_1[0], str_2[0], str_3[0]]]))
        poss = [[j for j in range(10)] for i in range(len(legend_base))] + [[0,1] for i in range(len(carry) - 1)]
        for i in heads:
            poss[i].remove(0)

        return cls({'legend': legend, 'cons': cons, 'poss': poss, 'solved': [None for i in range(len(legend))], 'tried': [[] for i in range(len(legend))],
                     'uniques': uniques
        , 'str1': str1, 'str2': str2, 'str3': str3, 'carry': carry})

    def __repr__(self):
        return f'Problem({self.legend}, {self.cons}, {self.poss}, {self.solved})'
    
    def add_tried(self, i, x):
        self.tried[i].append(x)
        return None
    

    # Solve for a variable given a value
    def solve(self, i, x):
        if i not in range(len(self.legend)) or x not in range(10):
            return False
        elif x in self.solved[:i] + self.solved[i+1:] and x in self.solved[:self.uniques] and i < self.uniques:
            return False
        elif self.solved[i] != None and self.solved[i] != x:
            return False
        self.solved[i] = x
        for j in range(len(self.cons)):
            if self.cons[j][i] != 0:
                self.cons[j][-1] += x*self.cons[j][i]
                self.cons[j][i] = 0
        if i < self.uniques:
            for j in range(self.uniques):
                if x in self.poss[j]:
                    self.poss[j].remove(x)
        self.poss[i] = [x]
        return True
    
    # Assume a variable is a value and solve for it
    def assume(self, i, x):
        self.assumptions.append((i, x))
        d = self.solve(i, x)
        return d


    # Check if there is a variable with only one possibility - if so, solves it
    def check_for_solved(self):
        x = []
        for i in range(len(self.legend)):
            if len(self.poss[i]) == 1 and self.solved[i] == None:
                self.solve(i, self.poss[i][0])
                #print(f'xSolved {self.legend[i]} to {self.poss[i][0]}')
                x.append((i, self.poss[i][0]))
        return x
    
    # Check if there is a variable with no possibilities - if so, returns True
    def check_for_impossible(self):
        for i in range(len(self.legend)):
            if len(self.poss[i]) == 0:
                return True
        return False
    
    # Check if all constraints are satisfied
    def check_constraints(self):
        for constraint in self.cons:
            unsolved = [i for i in range(len(constraint) - 1) if (constraint[i] != 0 and self.solved[i] == None)]
            if len(unsolved) == 0:
                inputs = [constraint[i]*self.solved[i] for i in range(len(constraint) - 1) if constraint[i] != 0]
                if sum(inputs) != -1*constraint[-1]:
                    return False
        return True

    def solve_single_cons(self, i):
        constraint = self.cons[i]
        x = [k for k, v in enumerate(constraint[:-1]) if v != 0]
        if len(x) == 1 and self.solved[x[0]] == None:
            self.log += f'Solved {self.legend[x[0]]} to {-1*int(constraint[-1]/constraint[x[0]])}\n'
            self.solve(x[0], -1*int(constraint[-1]/constraint[x[0]]))
            return (x[0], -1*int(constraint[-1]/constraint[x[0]]))
        else:
            return False

    
    # Check all constraints for the form ax = b and solves for x if so
    def solve_all_cons(self):
        newly_solved = []

        for i in range(len(self.cons)):
            x = self.solve_single_cons(i)
            if x:
                newly_solved.append(x)
        return newly_solved


    # for i in range(len(self.cons)):
    #         x = [k for k, v in enumerate(self.cons[i][:-1]) if v != 0]
    #         if len(x) == 1 and self.solved[x[0]] == None:
    #             print(f'Solved {self.legend[x[0]]} to {-1*int(self.cons[i][-1]/self.cons[i][x[0]])}')
    #             v = self.solve(x[0], -1*int(self.cons[i][-1]/self.cons[i][x[0]]))
    #             newly_solved.append((x[0], self.cons[i][-1]))


    # Loops solve_all_cons and check_for_solved until no more variables can be solved - most basic solve
    def basic_solve(self):
        x = self.check_for_solved() + self.solve_all_cons()
        while len(x) > 0:
            x = self.check_for_solved() + self.solve_all_cons()
        return self.check_for_impossible()


        
    #Take two constraints and unify them given a variable
    def unify_cons(self, i, j, x, return_cons=False):
        if i not in range(len(self.cons)) or j not in range(len(self.cons)) or x not in range(len(self.legend) - 1):
            return False
        elif self.cons[i][x] == 0 or self.cons[j][x] == 0:
            return False
        else: 
            self.cons[i][x] = int(self.cons[i][x])
            self.cons[j][x] = int(self.cons[j][x])
            lcm1 = lcm(self.cons[i][x], self.cons[j][x])
            new1 = [self.cons[i][k] * lcm1/self.cons[i][x] for k in range(len(self.cons[i]))]
            new2 = [self.cons[j][k] * lcm1/self.cons[j][x] for k in range(len(self.cons[j]))]
            new = [new1[k] - new2[k] for k in range(len(new1))]
            if return_cons:
                return new
            else:
                self.cons.append(new)
                return True
            
    # Take a constraint and a variable and restrict possible values for that variable
    def find_poss(self, i, x, max_variables=2, max_values=10, max_remaining=4):
        if x in self.legend:
            x = self.legend.index(x)
        if i not in range(len(self.cons)) or x not in range(len(self.legend)):
            self.error = 'Invalid constraint or variable'
            return False
        elif self.cons[i][x] == 0:
            self.error = 'Variable not in constraint'
            return False
        else:
            other_variables = [k for k in range(len(self.cons[i])-1) if k != x and self.cons[i][k] != 0]
            num_possibilities = prod([len(self.poss[k]) for k in other_variables])
            ov_unique = len([k for k in range(self.uniques) if k != x and self.cons[i][k] != 0])
            coeff = [self.cons[i][k] for k in other_variables]
            integer = self.cons[i][-1]
            z = self.cons[i][x]

            if len(other_variables) == 0:
                x_value = -1*int(self.cons[i][-1]/z)
                if x_value not in self.poss[x]:
                    self.error = 'No possible value for variable'
                    return False
                else:
                    self.log += f'Solved {self.legend[x]} to {x_value}\n'
                    self.solve(x, x_value)
                    return True
            elif len(other_variables) <= max_variables or num_possibilities <= max_values:
                poss = []
                for perm in itertools.product(*[self.poss[k] for k in other_variables]):
                    if len(list(set(perm[:ov_unique]))) == len(list(perm[:ov_unique])):
                        y = -1*int((integer + sum([coeff[k]*perm[k] for k in range(len(coeff))])))
                        if y % z == 0:
                            if int(y/z) in self.poss[x]:
                                poss.append(int(y/z))
                poss = list(set(poss))
                if len(poss) == 0:
                    self.poss[x] = []
                    # Add error message
                    self.error = 'No possible value for variable'
                    return False
                elif len(poss) <= max_remaining and len(poss) < len(self.poss[x]):
                    self.log += f'Using {self.cons_to_string(self.cons[i])} ' 
                    for k in other_variables + [x]:
                        self.log += f'{self.legend[k]} = {self.poss[k]} '
                    self.log += f'Calculated {self.legend[x]} to be {poss}\n'
                    self.poss[x] = list(set(poss))
                    if len(poss) == 1:
                        self.solve(x, poss[0])
                    return True
                else:
                    return True
            else:
                return True
    
    # Restrict possibilities for all variables based on single constraints
    def find_poss_all(self, round=1):
        max_variables, max_values, max_remaining = self.scheduler(round)
        last = self.poss.copy()
        for i, cons in enumerate(self.cons):
            for j in [i for i, x in enumerate(cons[:-1]) if x!= 0]:
                self.find_poss(i, j, max_variables, max_values, max_remaining)
            if self.check_for_impossible():
                return False
        self.check_for_solved()
        if last != self.poss:
            self.find_poss_all()
        elif round < self.max_rounds():
            self.find_poss_all(round + 1)
        return True
    
    def scheduler(self, rounds = 1):
        if rounds == 1:
            return (1,5,1)
        elif rounds == 2:
            return (1,5,2)
        elif rounds == 3:
            return (2,10,1)
        elif rounds == 4:
            return (2,10,2)
    
    def max_rounds(self):
        return 4
    
    # Generate unifications of constraints
    def generate_unifications(self, rounds=1):
        old_cons_num = 0
        for r in range(rounds):
            for i in range(len(self.cons)):
                for j in range(max(old_cons_num, i+1), len(self.cons)):
                    for x in [k for k in range(len(self.cons[i]) - 1) if self.cons[i][k] != 0 and self.cons[j][k] != 0]:
                        self.unify_cons(int(i), int(j), int(x))
            old_cons_num = len(self.cons)
        self.delete_repeat_cons()
    
    def is_multiple(self, cons1, cons2):
        if len(cons1) != len(cons2):
            return False
        else:
            non_zero_cons1 = [i for i in range(len(cons1)) if cons1[i] != 0]
            non_zero_cons2 = [i for i in range(len(cons2)) if cons2[i] != 0]
            if non_zero_cons1 == non_zero_cons2:
                for i, j in zip(cons1, cons2):
                    if i != 0 and j != 0 and i/j != cons1[non_zero_cons1[0]]/cons2[non_zero_cons2[0]]:
                        return False
                return True
            else:
                return False
    
    def delete_repeat_cons(self):
        array = []
        for cons1 in self.cons:
            x = True
            for cons2 in array:
                if self.is_multiple(cons1, cons2):
                    x = False
            if x:
                array.append(cons1)
        self.cons = array
        return True
                    


    def confirm_solution(self):
        for i in range(len(self.original_cons)):
            sum = [self.original_cons[i][k]*self.solved[k] for k in range(len(self.original_cons[i]) - 1)]
            if sum != -1*self.cons[i][-1]:
                return False
        return True
    
    def unify_cons_wrapped(self, i, j, x):
        if isinstance(x, str):
            x = self.legend.index(x)
        m = self.unify_cons(i, j, x)
        self.basic_solve()
        return m
                    
    def find_poss_wrapped(self, i, x):
        if isinstance(x, str):
            x = self.legend.index(x)
        m = self.find_poss(i, x)
        self.basic_solve()
        print(self.poss[x])
        return m

    def assume_wrapped(self, i, x):
        if isinstance(i, str):
            i = self.legend.index(i)
        m = self.assume(i, x)
        self.basic_solve()
        return m
    
    def cons_to_string(self, constraint):
        result = ''
        for i in range(len(constraint) - 1):
            if constraint[i] > 0:
                result += f'+ {constraint[i]}{self.legend[i]} '
            elif constraint[i] < 0:
                result += f'- {-1*constraint[i]}{self.legend[i]} '
        result += f'= {-1*constraint[-1]}'
        return result

    def show_state(self):
        result = 'Constraints:\n'
        for i, constraint in enumerate(self.cons):
            nz = [i for i in range(len(constraint) - 1) if constraint[i] != 0]
            if len(nz) > 0:
                result += f'{i}: '
                for i in range(len(constraint) - 1):
                    if constraint[i] > 0:
                        result += f'+ {constraint[i]}{self.legend[i]} '
                    elif constraint[i] < 0:
                        result += f'- {-1*constraint[i]}{self.legend[i]} '
                result += f'= {-1*constraint[-1]}'
                result += '\n'
        result += 'Possibilities:\n'
        for i in range(len(self.legend)):
            if self.solved[i] == None:
                result += f'{self.legend[i]}: {self.poss[i]}\n'
        #result += 'Solved:\n'
        for i in range(len(self.legend)):
            if self.solved[i] != None:
                pass
                #result += f'{self.legend[i]}: {self.solved[i]}\n'
        result += 'Assumptions:\n'
        for i in self.assumptions:
            result += f'{self.legend[i[0]]}: {i[1]}\n'

        table = [[('_' if self.solved[i] == None else self.solved[i]) if i!= None else '' for i in self.carry], 
        [(self.legend[i] if self.solved[i] == None else self.solved[i]) if i!= None else '' for i in self.str1],
        [(self.legend[i] if self.solved[i] == None else self.solved[i]) if i!= None else '' for i in self.str2],
        ['-'] * len(self.carry),
        [(self.legend[i] if self.solved[i] == None else self.solved[i]) if i!= None else '' for i in self.str3]]

        return result + tabulate(table)


class Solver_backtrack():

    def __init__(self, string):
         self.tree = [] 
         self.tree.append(Solver.from_crypt(string))
         self.log = []

    def __getattr__(self, __name: str) -> Any:
         return getattr(self.tree[-1], __name)
    
    def copy(self):

        # Deep copy all the mutable objects
        new_solver = Solver({
            'legend': copy.deepcopy(self.legend),
            'uniques': self.uniques,
            'cons': copy.deepcopy(self.cons),
            'poss': copy.deepcopy(self.poss),
            'solved': copy.deepcopy(self.solved),
            'tried': copy.deepcopy(self.tried),
            'carry': copy.deepcopy(self.carry),
            'str1': self.str1,
            'str2': self.str2,
            'str3': self.str3
        })
        new_solver.hidden_cons = copy.deepcopy(self.hidden_cons)
        new_solver.error = self.error
        new_solver.valid = self.valid
        new_solver.assumptions = copy.deepcopy(self.assumptions)
        new_solver.log = self.log
        return new_solver
   
    def make_assumption(self, i, x):
        try:
            # Create a deep copy of the current state
            new_state = self.tree[-1].copy()
            # Make the assumption in the new state
            new_state.assume_wrapped(i, x)
            # Push the new state to the stack
            self.tree.append(new_state)
            self.log.append(f'Assumed {new_state.legend[i]} = {x}')
        except Exception as e:
            self.log.append(f'Failed to assume {self.tree[-1].legend[i]} = {x} with error {str(e)}')

    
    def backtrack(self):
        if len(self.tree) > 1:
            # Pop the last state (failed assumption)
            failed_state = self.tree.pop()
            last_assumption = failed_state.assumptions[-1]
            self.log.append(f'Backtracked on assumption {failed_state.legend[last_assumption[0]]} = {last_assumption[1]}')
            return True
        return False

    def solve_with_backtracking(self):
        while self.tree:
            current_solver = self.tree[-1]
            if current_solver.basic_solve():
                if current_solver.check_for_impossible():
                    if not self.backtrack():
                        return False  # No solution found, and no more states to backtrack
                else:
                    return current_solver.solved  # Found a valid solution
            else:
                # Assume a new state based on the next possible heuristic or random choice
                # This should ideally be replaced by a more intelligent heuristic
                for i, poss in enumerate(current_solver.poss):
                    if len(poss) > 1:
                        self.make_assumption(i, poss[0])
                        break
        return False  # Exhausted all possibilities

    

# %% [markdown]
# TO DO:
# Display entire addition with solved digits
# 
# Display number before constraint
# 
# Final solved check
# 
# Add error messages
# 
# 
# Backtracking (Add wrapper)
# 
# Add Logging capabilities
# 


