import os
import json
import re
from solver import SolverNL  # Import the SolverNL class
from openai import OpenAI


MODEL_NAME = "gpt-4o"
PROBLEM = "EVEN + HERE = TENT"


# Load API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def ask_llm_for_instructions(solver_state):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You solve cryptarithmetic problems."},
            {"role": "user", "content": f"""
            Here are your previous actions taken, and the current state of the solver:

            {solver_state}

            Please provide the next set of instructions to solve the problem, formatted as a JSON list of strings.

            You have access to the following commands:
            - find_poss <constraint_index> <variable>: Find possible values for a variable based on a constraint. It will iterate through the possible values of other variables in the constraint assuming there aren't too many and return the possible values for the specified variable.
            - unify_cons <constraint_index_1> <constraint_index_2> <variable>: Unify two constraints based on a variable. It will combine the two equations and simplify them to create a new constraint with the specified variable removed.
            - make_assumption <variable> <value>: Assume a variable has a specific value. Use this when it doesn't seem like you are making progress on find_poss, or when you are sure that a value is a certain number.
            - backtrack: Undo the last assumption made and backtrack to the state before that assumption. Try and only use this when you are sure that the current path is incorrect.

            Do not use square brackets elsewhere in your response as they are used to denote the JSON list.

            For example, in order to  backtrack the previous assumption, find possible values for the letter E based on the constraint 0, unify constraints 1 and 2 based on the letter T, assume the letter E is equal to 5, you would provide the following instructions:

            [
                "backtrack",
                "find_poss 0 C2",
                "unify_cons 1 2 T",
                "make_assumption E 5"
            ]

            

            Follow the format above strictly as your instructions are parsed by a computer. Only the above command formats are supported. Do NOT write comments inside the instruction list.

            Think of the best set of instructions to provide based on the current state of the solver.
            Make sure that in find_poss and unify_cons, the variable is in the constraints. Feel free to assume something without backtracking in the same set of instructions so you can see what happens next. Feel free to think out loud before providing the instructions in a list. 

            An example of a log of a successful solve is:

            Constraints:
                0: - 1D + 1O + 1C1 = 0
                1: + 2D - 1O - 10C1 + 1C2 = 0
                2: + 1D + 1O - 1G - 10C2 = 0
                Possibilities:
                D: [1, 2, 3, 4, 5, 6, 7, 8, 9]
                O: [1, 2, 3, 4, 5, 6, 7, 8, 9]
                G: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                C1: [0, 1]
                C2: [0, 1]
                Assumptions:
                -  -  -
                _  _
                D  O
                O  D  D
                -  -  -
                D  O  G
                -  -  -

            Executed: find_poss 0 C1 - Success
            Executed: unify_cons 0 1 D - Success
            Executed: find_poss 3 O - Success
            Executed: find_poss 1 D - Success
            Assumed D = 7
            Executed: make_assumption D 7 - Success
            Backtracked on assumption D = 7
            Executed: backtrack - Success
            Assumed D = 8
            Executed: make_assumption D 8 - Success
            """}
        ]
    )
    return response.choices[0].message.content.strip()


def extract_instructions(response_text):
    try:
        # Use regex to find all JSON lists in the response text
        json_list_matches = re.findall(r'\[.*?\]', response_text, re.DOTALL)
        if json_list_matches:
            # Select the last JSON list found
            instructions_json = json_list_matches[-1]
            # Ensure that the extracted string is a valid JSON list
            instructions = json.loads(instructions_json)
            return instructions
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error extracting instructions: {str(e)}")
    return []


def main():
    # Define the problem
    problem = PROBLEM
    
    # Initialize the solver
    solver_llm = SolverNL(problem)
    
    count = 0

    # Loop until the problem is solved
    while not solver_llm.is_solved() and count < 10:

        # Get the current state of the solver
        solver_state = solver_llm.get_log_and_state()
        
        # Ask the LLM for instructions
        response_text = ask_llm_for_instructions(solver_state)

        print(f"Received instructions from LLM: {response_text}")
        
        # Extract and parse the instructions
        instructions_list = extract_instructions(response_text)
        
        if not instructions_list:
            print("No valid instructions received from LLM")
            break

        # Execute the instructions
        solver_llm.parse_and_execute_list(instructions_list)

        print(f"Executed instructions: {instructions_list}")
        print(f"Current state: {solver_llm.get_log_and_state()}")
        
        # Check if the problem is solved
        if solver_llm.is_solved():
            print("Problem solved!")
            break

        count += 1
    
    print("Problem Solved: ", solver_llm.is_solved())
    print("Rounds taken: ", count)

if __name__ == "__main__":
    main()
