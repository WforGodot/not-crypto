from typing import Dict, List, Tuple
from itertools import permutations


def invert_permutation(permutation: List[int]) -> Dict[int, int]:
    """
    Invert a permutation to map each value to its index.

    Args:
    permutation (List[int]): A list representing the permutation.

    Returns:
    Dict[int, int]: A dictionary mapping each value in the permutation to its index.
    """
    return {value: index for index, value in enumerate(permutation)}


def generate_permutations(digits: List[List[int]]) -> List[Tuple[List[List[int]], Dict[int, str]]]:
    """
    Generate all valid permutations for the given list of lists of digits,
    avoiding permutations where the first digit of any list element is zero.

    Args:
    digits (List[List[int]]): A list of lists where each sublist represents a set of digits.

    Returns:
    List[Tuple[List[List[int]], Dict[int, str]]]: A list of tuples where each tuple contains a
    permuted version of the digits and a dictionary mapping the original digits to their
    corresponding characters based on the permutation.
    """
    max_digit = max([max(sublist) for sublist in digits])
    first_digits = [sublist[0] for sublist in digits]
    valid_permutations = []
    
    for permutation in permutations(range(10), max_digit + 1):
        if 0 not in [permutation[i] for i in first_digits]:  # Ensure no leading zero in any sublist
            translated = [[permutation[digit] for digit in sublist] for sublist in digits]
            mapping = invert_permutation(list(permutation))
            valid_permutations.append((translated, {digit: chr(97 + i) for i, digit in enumerate(mapping)}))
    
    return valid_permutations
