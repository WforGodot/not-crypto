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
            self.error = 'Invalid variable or value'
            return False
        elif x in self.solved[:i] + self.solved[i+1:] and x in self.solved[:self.uniques] and i < self.uniques:
            self.error = 'Value already used'
            return False
        elif self.solved[i] != None and self.solved[i] != x:
            self.error = 'Variable already solved'
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

    # Assume variable i is x and solve for it
    def assume(self, i, x):
        if isinstance(i, str):
            try:
                i = self.legend.index(i)
            except:
                self.error = 'Invalid letter'
                return False
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
        if all([len(self.poss[i]) == 1 for i in range(len(self.legend))]) and not self.is_solved():
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
            value = -1*int(constraint[-1]/constraint[x[0]])
            self.log += f'Solved {self.legend[x[0]]} to {value}\n'
            self.solve(x[0], value)
            return (x[0], value)

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
        if self.check_for_impossible():
            return False
        if self.is_solved():
            return True
        x = self.check_for_solved() + self.solve_all_cons()
        count = 0
        while len(x) > 0 and count < 5:
            x = self.check_for_solved() + self.solve_all_cons()
            count += 1
        return not self.check_for_impossible()

    #Take two constraints and unify them given a variable
    def unify_cons(self, i, j, x, return_cons=False):
        if i not in range(len(self.cons)) or j not in range(len(self.cons)) or x not in range(len(self.legend) - 1):
            self.error = 'Invalid constraint or variable'
            return False
        elif self.cons[i][x] == 0 or self.cons[j][x] == 0:
            self.error = 'Variable not in constraint'
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
    def find_poss(self, i, x, max_variables=3, max_values=30, max_remaining=10, max_starting=3):
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

            for k in other_variables:
                if not self.poss[k]:
                    self.error = 'No possible value for variable'
                    return False

            # Boundary conditions (Find min and max possible values based on min and max values of other variables)
            min_value = -((integer + sum([max(map(lambda val: self.cons[i][k] * val, self.poss[k])) for k in other_variables])) // z)
            max_value = -((integer + sum([min(map(lambda val: self.cons[i][k] * val, self.poss[k])) for k in other_variables])) // z)

            # Adjust for the sign of z
            if z < 0:
                min_value, max_value = max_value, min_value

            if min_value > max_value:
                self.error = 'No possible value for variable'
                return False

            # Restrict possible values for x within the calculated range
            self.poss[x] = [val for val in self.poss[x] if min_value <= val <= max_value]

            # Ensure that poss[x] is not empty
            if not self.poss[x]:
                self.error = 'No possible value for variable'
                return False

            if len(other_variables) == 1 or (len(other_variables) <= max_variables and num_possibilities <= max_values):
                poss = []
                for perm in itertools.product(*[self.poss[k] for k in other_variables]):
                    if len(list(set(perm[:ov_unique]))) == len(list(perm[:ov_unique])):
                        y = -1*int((integer + sum([coeff[k]*perm[k] for k in range(len(coeff))])))
                        if y % z == 0:
                            if int(y/z) in self.poss[x]:
                                if not int(y/z) in perm[:ov_unique]:
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
    def find_poss_all(self, round=1, hacks=False):
        max_variables, max_values, max_remaining = self.scheduler(round)
        last = self.poss.copy()
        for i, cons in enumerate(self.cons):
            for j in [i for i, x in enumerate(cons[:-1]) if x != 0]:
                self.find_poss(i, j, max_variables, max_values, max_remaining)
            if self.check_for_impossible():
                return False
        self.basic_solve()
        self.check_for_solved()
        if last != self.poss:
            self.find_poss_all()
        elif round < self.max_rounds(hacks):
            self.find_poss_all(round + 1)
        return True

    def scheduler(self, rounds=1):
        if rounds == 1:
            return (1, 10, 10)
        elif rounds == 2:
            return (2, 20, 1)
        elif rounds == 3:
            return (2, 20, 5)
        elif rounds == 4:
            return (2, 100, 10)

    def max_rounds(self, hacks):
        if hacks:
            return 4
        else:
            return 1
    
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
            sum1 = sum([self.original_cons[i][k]*self.solved[k] for k in range(len(self.original_cons[i]) - 1)])
            if sum1 != -1*self.cons[i][-1]:
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
        return m

    def assume_wrapped(self, i, x):
        if isinstance(i, str):
            try:
                i = self.legend.index(i)
            except:
                self.error = 'Invalid letter'
                return False
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

    def show_state(self, string=False):
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



        if not string:
            return result + tabulate(table)
        else:
            return result + "Equation: " + self.get_equation()

    def get_equation(self):
        # Construct the equation string
        def sub_in_values(lst):
            return ''.join([str(self.solved[val]) if val is not None and self.solved[val] is not None else self.legend[val] if val is not None else '' for val in lst])

        equation = f"{sub_in_values(self.str1)} + {sub_in_values(self.str2)} = {sub_in_values(self.str3)}"
        return equation

    
    def is_solved(self):
        if all([i != None for i in self.solved]):
            return self.confirm_solution()
        return False
    

    def make_copy(self):

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


class Solver_backtrack():

    def __init__(self, string):
        self.tree = [] 
        self.tree.append(Solver.from_crypt(string))
        self.log = []

        self.answer = None

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.tree[-1], __name)
    

   
    def make_assumption(self, i, x):
        if isinstance(i, str):
            try:
                i = self.tree[-1].legend.index(i)
            except Exception as e:
                self.log.append('Invalid letter')
                return False
        try:
            # Create a deep copy of the current state
            new_state = self.tree[-1].make_copy()
            # Make the assumption in the new state
            new_state.assume_wrapped(i, x)
            # Push the new state to the stack
            self.tree.append(new_state)

            

            return new_state
        except Exception as e:
            self.log.append(f'Failed to assume {i} = {x} with error {str(e)}')
 
    def backtrack(self):
        if len(self.tree) > 1:
            # Pop the last state (failed assumption)
            failed_state = self.tree.pop()
            if failed_state.assumptions:
                last_assumption = failed_state.assumptions[-1]
                
            return True
        return False

    def solve_with_backtracking(self, current_solver=None):
        if current_solver is None:
            current_solver = self.tree[-1]
        current_solver.find_poss_all(hacks=True)

        if self.answer:
            return True
        if current_solver.is_solved():
            current_solver.show_state()
            self.answer = current_solver
            return True
        
        possibilities = [(i, p) for i, p in enumerate(current_solver.poss) if len(p) > 1]
        if not possibilities:
            return False  # No more possibilities and not solved

        index, values = min(possibilities, key=lambda x: len(x[1]))
        for value in values:
            new_state = current_solver.make_copy()
            new_state.assume_wrapped(index, value)
            if self.solve_with_backtracking(new_state):
                return True  # If a solution is found, return True

        return False  # If none of the values lead to a solution
    

class SolverNL(Solver_backtrack):

    def __init__(self, string):
        super().__init__(string)
        self.log = []
    
    def parse_and_execute(self, instruction):
        try:
            # Split instruction into command and arguments
            parts = instruction.strip().split()
            command = parts[0]
            args = parts[1:]

            equation = self.tree[-1].get_equation()
            
            message = []
            # Execute the command
            if command == 'find':
                if len(args) == 2:
                    i = int(args[0])
                    x = args[1]
                    if x in self.tree[-1].legend:
                        x = self.tree[-1].legend.index(x)
                    before_poss = copy.deepcopy(self.tree[-1].poss[x])
                    success = self.find_poss_wrapped(i, x)
                    if success:
                        after_poss = self.tree[-1].poss[x]
                        removed = set(before_poss) - set(after_poss)
                        if removed:
                            message.append(f"Reduced possibilities for {self.tree[-1].legend[x]}. Remaining: {after_poss}, Removed: {list(removed)}")
                        else:
                            message.append(f"No possibilities were reduced for {self.tree[-1].legend[x]}.")
                else:
                    raise ValueError("find requires 2 arguments")
            elif command == 'unify':
                if len(args) == 3:
                    i = int(args[0])
                    j = int(args[1])
                    x = args[2]
                    new_constraint = self.tree[-1].unify_cons(i, j, self.tree[-1].legend.index(x), return_cons=True)
                    success = self.unify_cons_wrapped(i, j, x)
                    if success:
                        parsed_constraint = self.tree[-1].cons_to_string(new_constraint)
                        message.append(f'Unified constraints into: {parsed_constraint}')
                else:
                    raise ValueError("unify requires 3 arguments")
            elif command == 'assume':
                if len(args) == 2:
                    i = args[0]
                    if x in self.tree[-1].legend:
                        x = self.tree[-1].legend.index(x)
                    x = int(args[1])
                    success = self.make_assumption(i, x)
                    if success:
                        message.append(f'Assumed {self.tree[-1].legend[x]} = {i}')
                else:
                    raise ValueError("assume requires 2 arguments")
            elif command == 'backtrack':
                x = self.tree[-1].assumptions[-1] if self.tree[-1].assumptions else None   
                success = self.backtrack()
                if success:
                    message.append(f'Backtracked on assumption {x[0]} = {x[1]}')
                if not success:
                    raise ValueError("Cannot backtrack further")
            else:
                raise ValueError("Unknown command")
            
            # Log the action and result
            if success:
                self.log.append(f"Executed: {instruction} - Success")
                self.log += message
            else:
                self.log.append(f"Executed: {instruction} - Failed: {self.tree[-1].error}")

            # Log the new equation if it has changed
            new_equation = self.tree[-1].get_equation()
            if new_equation != equation:
                self.log.append(f"New equation: {new_equation}")

            return success

        except Exception as e:
            self.log.append(f"Executed: {instruction} - Error: {str(e)}")
            return False
    
    def parse_and_execute_list(self, instructions):
        for instruction in instructions:
            self.parse_and_execute(instruction)
            self.find_poss_all()
            if self.is_solved():
                self.log.append("Problem Solved!")
            if self.check_for_impossible():
                self.log.append("The current state with assumptions: " +
                            str([(self.legend[x[0]], x[1]) for x in self.assumptions]) +
                            " is impossible. Please backtrack and try again.")

                            
    def get_log_and_state(self):
        current_state = self.tree[-1].show_state(string=True)
        return '\n'.join(self.log) + "\n\nCurrent State:\n" + current_state

    
     
if __name__ == '__main__':
    s = Solver_backtrack('EYE + TEEN = MONTH')
    s.solve_with_backtracking()
    print(s.solved.show_state())

    




