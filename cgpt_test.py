import os
import random
import json
from openai import OpenAI

# Set the number of questions to ask GPT
NUM_QUESTIONS = 5
QUESTION_PATH = r"C:\Users\proje\Documents\GitHub\not-crypto\questions\test.json"


# Load API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def ask_gpt(question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpful assistant. You solve math problems."},
            {"role": "user", "content": "Solve the given cryptarithmetic problem and \
             provide your final answer on the right side of the equality in \
             double square brackets. Eg. The answer is [[4221]]. Question: " + question}
        ]
    )
    return response.choices[0].message.content.strip()


def main():
    # Load the questions from the JSON file
    with open(QUESTION_PATH, 'r') as f:
        questions = json.load(f)

    # Randomly select NUM_QUESTIONS questions
    selected_questions = random.sample(questions, NUM_QUESTIONS)

    results = []
    correct_count = 0

    for idx, item in enumerate(selected_questions):
        question = item["Question"]
        correct_answer = item["Answer"].split("=")[1].strip()
        
        # Ask GPT the question
        gpt_response = ask_gpt(question)
        
        # Extract the answer from the GPT response
        start = gpt_response.find("[[") + 2
        end = gpt_response.find("]]")
        gpt_answer = gpt_response[start:end].strip()
        
        # Compare GPT's answer to the correct answer
        is_correct = gpt_answer == correct_answer
        if is_correct:
            correct_count += 1
        
        # Log the result
        results.append({
            "Question": question,
            "GPT_Response": gpt_response,
            "GPT_Answer": gpt_answer,
            "Correct_Answer": correct_answer,
            "Is_Correct": is_correct
        })
        
        # Print the number of correct answers every 10 questions
        if (idx + 1) % 10 == 0:
            print(f"{correct_count}/{idx + 1} questions correct")

    # Save the results to a JSON file
    with open('results.json', 'a') as f:
        json.dump(results, f, indent=4)

    print(f"Final score: {correct_count}/{NUM_QUESTIONS}")

if __name__ == "__main__":
    main()
