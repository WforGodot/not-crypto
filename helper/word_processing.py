
def clean_word(word: str) -> str:
    """
    Cleans a word by removing punctuation and other special characters.

    Parameters:
    - word (str): The word to be cleaned.

    Returns:
    - str: The cleaned word, with all special characters removed.
    """
    for char in "' ?.!/;:&":
        word = word.replace(char, '')
    return word


def duplicate_letters(word_list: list) -> list:
    """
    Generates a new list of words by duplicating each character in each word of the input list.
    This helps in creating potential matches for cryptarithmetic puzzles where duplicated letters
    might lead to correct solutions.

    Parameters:
    - word_list (list): The list of words to process.

    Returns:
    - list: A new list of words with each character duplicated.
    """
    duplicated_words = []
    for word in word_list:
        new_word = ""
        for char in word:
            new_word += char * 2  # Duplicate each character
        duplicated_words.append(new_word)
    return duplicated_words
