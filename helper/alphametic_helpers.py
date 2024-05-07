from typing import List, Dict, Tuple, Any
from collections import defaultdict
import random


def to_result(permutation: Tuple[List[List[int]], Dict[int, int]]) -> str:
    """Generate a string representation of the result from a given permutation mapping."""
    summands, number_mapping = permutation
    total = sum(int(''.join(map(str, numbers))) for numbers in summands)
    result_str = list(map(int, str(total)))
    translated = []
    additional_mapping = {}
    next_char = 97  # ASCII for 'a'
    
    for digit in result_str:
        if digit in number_mapping:
            translated.append(number_mapping[digit])
        elif digit in additional_mapping:
            translated.append(additional_mapping[digit])
        else:
            additional_mapping[digit] = chr(next_char)
            translated.append(chr(next_char))
            next_char += 1
            
    return ''.join(map(str, translated))


def select_permutations(permutations: List[Tuple[List[List[int]], Dict[int, int]]]) -> List[List[int]]:
    """Select unique alphametic solutions from a list of permutations."""
    categorized = categorize_by_function(permutations, to_result)
    unique_permutations = [values[0] for key, values in categorized.items() if len(values) == 1]
    summands = [[int(''.join(map(str, numbers))) for numbers in perm[0]] for perm in unique_permutations]
    summands_with_total = [numbers + [sum(numbers)] for numbers in summands]
    return summands_with_total


def encode_alphametic(alphametic: List[List[int]]) -> str:
    """Encode alphametic solutions into a string format that can be used for comparison."""
    numbers_as_strings = [str(num) for num in alphametic]
    max_length = max(len(num_str) for num_str in numbers_as_strings)
    padded_numbers = [num_str.zfill(max_length) for num_str in numbers_as_strings]
    return ''.join(sorted((nums[i] for nums in padded_numbers)) for i in range(max_length))


def categorize_by_function(items: List[Any], func: Any) -> Dict[str, List[Any]]:
    """Categorize items by a function that applies to each item."""
    result = defaultdict(list)
    for item in items:
        key = func(item)
        result[key].append(item)
    return dict(result)


def find_letters(numbers: List[str], current_mapping: Dict[str, str], word_dict: Dict[str, List[str]]) -> Tuple[Dict[str, str], bool]:
    if not numbers:
        return current_mapping, True  # Base case: all numbers processed successfully

    number_encode = decompose(numbers[0])
    if number_encode in word_dict:
        for word in word_dict[number_encode]:
            updated_mapping = match_number_to_word(numbers[0], word, current_mapping)
            if updated_mapping:
                # If this is the last number, return the mapping immediately
                if len(numbers) == 1:
                    return updated_mapping, True
                # Recurse with the rest of the numbers
                subsequent_mapping, perfect_match = find_letters(numbers[1:], updated_mapping, word_dict)
                if perfect_match:
                    return subsequent_mapping, True  # Return first complete valid mapping found

    # If no valid mappings were found for this number, and there are more numbers to try, continue with the next one
    if len(numbers) > 1:
        return find_letters(numbers[1:], current_mapping, word_dict)

    return {}, False  # No valid mappings found and no numbers left to skip



def match_number_to_word(number: str, word: str, current_map: Dict[str, str]) -> Dict[str, str]:
    """Attempt to match a single number to a word using a given mapping, ensuring consistency."""
    updated_map = current_map.copy()
    for num_char, word_char in zip(number, word):
        if num_char in updated_map:
            if updated_map[num_char] != word_char:
                return None  # Mismatch in mapping
        elif word_char in updated_map.values():
            return None  # Reverse mapping conflict
        updated_map[num_char] = word_char
    return updated_map


def match_number_to_word(number: str, word: str, current_map: Dict[str, str]) -> Dict[str, str]:
    """Attempt to match a single number to a word using a given mapping."""
    updated_map = current_map.copy()

    for num_char, word_char in zip(number, word):
        if num_char in updated_map:
            if updated_map[num_char] != word_char:
                return None
        elif word_char in updated_map.values():
            return None
        updated_map[num_char] = word_char
    return updated_map


def sort_numbers_by_frequency(numbers: List[str], frequency_dict: Dict[str, int]) -> List[str]:
    """Sort numbers based on the frequency of their decomposed values appearing in a frequency dictionary."""
    return sorted(numbers, key=lambda num: frequency_dict.get(decompose(num), 0))

def sort_numbers_by_word_dict_frequency(numbers: List[str], word_dict: Dict[str, List[str]]) -> List[str]:
    """
    Sort numbers based on the frequency of their decomposed values appearing in a frequency dictionary.

    Args:
        numbers (List[str]): List of numbers as strings.
        word_dict (Dict[str, List[str]]): A dictionary of words categorized by decomposed indices, where each key is a decomposed index sequence and each value is a list of words matching that sequence.

    Returns:
        List[str]: List of numbers sorted based on the frequency of their decomposed values in the word dictionary.
    """
    frequency_dict = {decompose(num): len(word_dict.get(decompose(num), [])) for num in numbers}
    return sorted(numbers, key=lambda num: frequency_dict.get(decompose(num), 0), reverse=True)

# You would need to import this function where needed or define it in the relevant module if it's part of a larger package.



def parse_mapping_to_words(mappings: List[Dict[str, str]], numbers: List[str], sorted_numbers: List[str]) -> str:
    """
    Parse multiple mappings of numbers to alphabetic characters into word-like forms.

    Args:
        mappings (List[Dict[str, str]]): List of mappings from numbers to letters.
        numbers (List[str]): Original list of numbers as strings.
        sorted_numbers (List[str]): Sorted list of numbers used for mapping.

    Returns:
        str: A string representing the alphametic words formed from the numbers using the provided mappings.
    """

    all_alphametic_words = []
    for mapping in mappings:
        alphametic_words = []
        for number in numbers:
            alphametic_words.append(''.join(mapping.get(char, '?') for char in number))
        all_alphametic_words.extend(alphametic_words)

    return ' + '.join(all_alphametic_words)


def decompose(number: str) -> str:
    """
    Decompose a number into its unique set of digits with corresponding indexes.
    This function maps each unique digit to the index of its first appearance in the number.

    Args:
        number (str): The number as a string, from which to decompose digits.

    Returns:
        str: A string of indices representing each digit's first occurrence.
    """
    unique_chars = []  # Use a list to maintain the order of characters
    index_map = {}  # Dictionary to keep track of first occurrences
    indexes = ''

    for char in number:
        if char not in index_map:
            index_map[char] = len(unique_chars)
            unique_chars.append(char)
        indexes += str(index_map[char])

    return indexes
