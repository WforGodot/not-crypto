from typing import Dict, List, Tuple, Any


def categorize_by_function(data: List[Any], function: callable) -> Dict[str, List[Any]]:
    """
    Categorizes a list of items based on a specified function.

    Args:
    data (List[Any]): A list of items to categorize.
    function (callable): A function that applies to items to derive a key for categorization.

    Returns:
    Dict[str, List[Any]]: A dictionary where each key is the result of the function and
                           the value is a list of items that share the same key.
    """
    categorized_result = {}
    for item in data:
        key = function(item)
        if key in categorized_result:
            categorized_result[key].append(item)
        else:
            categorized_result[key] = [item]
    return categorized_result


def decompose(sequence: List[Any]) -> Tuple[List[Any], str]:
    """
    Creates a unique representation of the sequence by mapping each unique item
    to its first occurrence index, returning the mapped values and the sequence
    transformed into these indices.

    Args:
    sequence (List[Any]): The sequence to decompose.

    Returns:
    Tuple[List[Any], str]: A tuple containing the list of unique items and
                            a string representing the sequence by indices.
    """
    unique_items = []
    indices = []
    item_to_index = {}
    for item in sequence:
        if item not in item_to_index:
            item_to_index[item] = len(unique_items)
            unique_items.append(item)
        indices.append(str(item_to_index[item]))
    return unique_items, ''.join(indices)


def compose(values: List[Any], indices: str) -> List[Any]:
    """
    Reconstructs the sequence from the mapped values and indices.

    Args:
    values (List[Any]): The list of unique items.
    indices (str): The string of indices representing the sequence.

    Returns:
    List[Any]: The original sequence reconstructed from values and indices.
    """
    return [values[int(index)] for index in indices]


def decompose_to_index_sequence(sequence: List[Any]) -> str:
    """
    Converts a sequence to a string of indices based on first occurrence of elements.

    Args:
    sequence (List[Any]): The sequence to convert.

    Returns:
    str: The sequence represented by indices.
    """
    _, index_sequence = decompose(sequence)
    return index_sequence

