from typing import List


def update_list(existing_lists: List[List[int]]) -> List[List[int]]:
    """
    Expands each list in the existing_lists by appending a new number ranging from 0 to one 
    more than the maximum element in the list. This expansion helps in generating permutations 
    with increased elements.

    Args:
        existing_lists (List[List[int]]): A list containing multiple lists of integers.

    Returns:
        List[List[int]]: An expanded list of lists, where each sublist is extended by 
                         appending additional numbers.
    """
    new_lists = []
    for sublist in existing_lists:
        max_element = max(sublist)
        new_lists.extend([sublist + [i] for i in range(max_element + 2)])
    return new_lists


def generate_list(target_length: int) -> List[List[int]]:
    """
    Generates all permutations of lists with incremental numbers starting from 0 up to 
    target_length - 1. This is achieved by recursively expanding the lists from a single element.

    Args:
        target_length (int): The desired length of the list of numbers.

    Returns:
        List[List[int]]: A list of all possible number lists of specified length.
    """
    current_lists = [[0]]
    for _ in range(target_length - 1):
        current_lists = update_list(current_lists)
    return current_lists


def split_list(original_list: List[int], lengths: List[int]) -> List[List[int]]:
    """
    Splits a list into sublists of specified lengths. This function is useful for partitioning 
    a list based on predefined segment sizes.

    Args:
        original_list (List[int]): The list to be split.
        lengths (List[int]): A list of integers representing the lengths of each segment 
                             to be created from the original list.

    Returns:
        List[List[int]]: A list where each element is a sublist of the original list 
                         corresponding to the specified lengths.
    """
    sublists = []
    start_index = 0
    for length in lengths:
        end_index = start_index + length
        sublists.append(original_list[start_index:end_index])
        start_index = end_index
    return sublists


def generate_numbers(lengths: List[int]) -> List[List[List[int]]]:
    """
    Generates a list of numbers split into sublists according to specified lengths. 
    This is used to create structured data for problems such as generating alphametics.

    Args:
        lengths (List[int]): A list of integers where each integer specifies the length of 
                             each sublist that should be generated.

    Returns:
        List[List[List[int]]]: A list where each element is a list of lists of integers, 
                                split according to the specified lengths.
    """
    list_sum = sum(lengths)
    list_of_lists = generate_list(list_sum)
    list_of_lists = [split_list(list1, lengths) for list1 in list_of_lists]
    return list_of_lists
