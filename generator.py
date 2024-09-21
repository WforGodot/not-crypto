from typing import List, Tuple, Dict
import json
from helper.word_processing import clean_word, duplicate_letters
from helper.alphametic_helpers import find_letters, parse_mapping_to_words, select_permutations, sort_numbers_by_word_dict_frequency
from helper.categorize import categorize_by_function, decompose_to_index_sequence
from helper.permutation_utils import generate_permutations
from helper.list_manipulation import generate_numbers


def load_words(file_path: str) -> List[str]:
    """
    Load words from a file and return a list of cleaned words.

    Args:
        file_path (str): The file path to read words from.

    Returns:
        List[str]: A list of cleaned and uppercased words.
    """
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return [clean_word(word.upper()) for word in words]


def generate_extended_word_list(common_words: List[str], all_words: List[str]) -> List[str]:
    """
    Generate an extended list of words by including triple repeated characters.

    Args:
        common_words (List[str]): List of common words.
        all_words (List[str]): List of all words.

    Returns:
        List[str]: The extended list of words.
    """
    words = common_words + all_words
    extras = [chr(97 + i) * 3 for i in range(26)]
    return words + extras


def prepare_word_dictionaries(words: List[str], max_length: int, duplit: bool = False) -> Dict[str, List[str]]:
    """
    Prepare dictionaries of words categorized by their decomposed indices. Optionally include only words with duplicated letters.

    Args:
        words (List[str]): List of words to process.
        max_length (int): Maximum length of words to include in the dictionary.
        duplit (bool): If True, only include words with duplicated letters; otherwise, include all words.

    Returns:
        Dict[str, List[str]]: A dictionary of words categorized by their decomposed indices.
    """
    if duplit:
        words = duplicate_letters(words)  # Filter to include only words with duplicated letters

    filtered_words = [word for word in words if len(word) <= max_length]
    unique_words = list(dict.fromkeys(filtered_words))  # Remove duplicates and maintain order
    return categorize_by_function(unique_words, decompose_to_index_sequence)

def get_alphametics(lengths: List[int]) -> List[List[int]]:
    """
    Generate alphametic solutions for given lengths of numbers.
    """
    alphametics = [select_permutations(generate_permutations(numbers)) for numbers in generate_numbers(lengths)]
    flattened_alphametics = [item for sublist in alphametics for item in sublist]
    return flattened_alphametics


def find_alphametic(numbers: List[int], word_dict: Dict[str, List[str]]) -> Tuple[str, str]:
    """
    Find alphametic solutions for a set of numbers using provided word dictionaries.

    Args:
        numbers (List[int]): A list of integers to find solutions for.
        word_dict (Dict[str, List[str]]): A dictionary of words categorized by decomposed indices.

    Returns:
        Tuple[str, str]: A tuple of question and answer format for alphametics.
    """
    str_numbers = [str(num) for num in numbers]
    sorted_numbers = sort_numbers_by_word_dict_frequency(str_numbers, word_dict)

    mapping, perfect = find_letters(sorted_numbers, {}, word_dict)  # Adjusted to use the new return type of find_letters

    if not perfect:
        return "No valid mapping found", ""

    # If a valid mapping is found, generate the question format.
    question_format = parse_mapping_to_words([mapping], str_numbers, sorted_numbers)

    # Generate answer format as a simple equation of the sum.
    total = sum(numbers)
    answer_format = ' + '.join(str_numbers) + ' = ' + str(total)

    return question_format, answer_format


def main():
    # Load words
    common_words = load_words('wordlist/common_words.txt')
    all_words = load_words('wordlist/wordlist.txt')
    
    # Generate extended word list
    words = generate_extended_word_list(common_words, all_words)
    
    # Prepare word dictionaries
    common_word_dict = prepare_word_dictionaries(common_words, 7)
    #normal_word_dict = prepare_word_dictionaries(words, 7)
    #duplit_dict = prepare_word_dictionaries(words, 7, duplit=True)

    # Example usage: Generate alphametics
    lengths = [2, 3]  # Lengths of numbers in the alphametic puzzle
    alphametics = get_alphametics(lengths)

    results = []  # To store both questions and answers in desired format

    # Process each generated alphametic to find those with valid word encodings
    for numbers in alphametics:
        question_format, answer_format = find_alphametic(numbers, common_word_dict)
        
        if question_format and answer_format:  # Only include if both questions and answers are non-empty
            if '?' not in question_format and '?' not in answer_format:
                results.append({"Question": question_format, "Answer": answer_format})

    # Save results to a JSON file
    with open('output.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
