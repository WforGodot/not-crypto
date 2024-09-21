# %%

import json
from itertools import permutations
# %%
from collections import defaultdict

def update_list(input_list):
    """
    Generate a list of lists of length n from input_list
    """
    output_list = []
    for list1 in input_list:
        max1 = max(list1)
        output_list = output_list + [list1 + [i] for i in range(0,max1+2)]
    return output_list

def generate_list(n):
    """
    Generate a list of lists of length n
    """
    list1 = [[0]]
    for i in range(0,n-1):
        list1 = update_list(list1)
    return list1

def split_list(lst, lengths):
    sublists = []
    start = 0
    for length in lengths:
        # Take a slice of the original list with the specified length
        sublists.append(lst[start:start + length])
        start += length
    return sublists

# %%


def generate_numbers(lengths):
    
    list_sum = sum(lengths)
    list_of_lists = generate_list(list_sum)
    list_of_lists = [split_list(list1, lengths) for list1 in list_of_lists]
    return list_of_lists    

def invert_permutation(permutation):
    return {value: index for index, value in enumerate(permutation)}


def generate_permutations(list1):
    max1 = max([max(list) for list in list1])
    fst = [list[0] for list in list1]
    perms = []
    for p in permutations(range(10), max1+1):
        if 0 in [p[i] for i in fst]:
            pass
        else:
            translated = [[p[i] for i in list] for list in list1]
            dict1 = invert_permutation(p)
            perms.append((translated, dict1))
    return perms



def categorize_by_f(long_list, f):
    result = defaultdict(list)
    for x in long_list:
        result[f(x)].append(x)
    
        
    return dict(result)
    
def to_result(perm):
    summands, mapping = perm
    sum1 = sum([int(''.join(map(str, numbers))) for numbers in summands])
    sum1 = list(map(int, str(sum1)))
    translated = []
    counter = 97
    new = {}
    for digit in sum1:
        if digit in mapping.keys():
            translated.append(mapping[digit])
        elif digit in new.keys():
            translated.append(new[digit])
        else:
            new[digit] = chr(counter)
            translated.append(chr(counter))
            counter += 1
    return ''.join(map(str, translated))

def select_permutations(perms):

    x = categorize_by_f(perms, to_result)
    z = [v[0] for k, v in x.items() if len(v) == 1]
    summands = [[int(''.join(map(str, numbers)))for numbers in y] for y, _ in z]
    summands = [x + [sum(x)] for x in summands]
    return summands


# %%
def get_alphametics(lengths):

    alphametics = [select_permutations(generate_permutations(perm)) for perm in generate_numbers(lengths)]
    alphametics = [item for sublist in alphametics for item in sublist]
    return alphametics

# %%
def pad(x, n):
    return '0' * (n - len(x)) + x

def pick(alphametic, digit):
    x = sorted([x[digit] for x in alphametic[:-1]]) + [alphametic[-1][digit]]
    return ''.join(x)
    

def encode_alphametic(alphametic):
    alphametic = [str(x) for x in alphametic]
    length = max(len(x) for x in alphametic)
    padded = [pad(x, length) for x in alphametic]
    return ''.join([pick(padded, i) for i in range(length)])

def categorize_alphametics(alphametics):
    return categorize_by_f(alphametics, encode_alphametic)

# %%
metics = get_alphametics([3,3])
print([v[0] for v in categorize_alphametics(metics).values()])

# %%
def fix_word(word):
    for char in word:
        if char in "' ?.!/;:&":
            word = word.replace(char,'')
    return word

def duplit(word_list):
    words = word_list.copy()
    for word in word_list:
        for i in range(len(word)):
            words.append(word[:i]+word[i]+word[i:])
    return words

# %%
with open('common_words.txt', 'r') as file:
    common_words = file.read().split('\n')
with open('wordlist.txt', 'r') as file:
    all_words = file.read().split('\n')

common_words = [fix_word(word.upper()) for word in common_words]
all_words = [fix_word(word.upper()) for word in all_words]
words = common_words + all_words
extras = [chr(97+i) for i in range(26)]
extras = [i+i+i for i in extras]           
words = words + extras

duplit_words = duplit(words)
duplit_words = list(filter(lambda x: len(x) < 7, duplit_words))
duplit_words = list(dict.fromkeys((duplit_words)))

with open('common_words_parsed.txt', 'w') as file:
    for word in common_words:
        file.write(word + '\n')
with open('words_parsed.txt', 'w') as file:
    for word in words:
        file.write(word + '\n')
with open('duplit_words_parsed.txt', 'w') as file:
    for word in duplit_words:
        file.write(word + '\n')


def decompose(x):
    values = list(dict.fromkeys((x)))
    return values, ''.join([str(values.index(v)) for v in x])
def compose(values, x):
    return ''.join([values[int(v)] for v in x])
def decompose2(x):
    return decompose(x)[1]

common_word_dict = categorize_by_f(common_words, decompose2)
normal_word_dict = categorize_by_f(words, decompose2)
duplit_dict = categorize_by_f(duplit_words, decompose2)


# %%
def find_letters(numbers, map, dict1):
    if len(numbers) == 0:
        return [map], True

    encode = decompose2(numbers[0])

    maps = []
    if encode in dict1:
        for word in dict1[encode]:
            x = match(numbers[0], word, map)
            if x:
                y, perfect = find_letters(numbers[1:], x, dict1)
                if perfect:
                    maps+=y
    if len(maps) > 0:
        return maps, True
    else:
        z, perfect = find_letters(numbers[1:], map, dict1)

    return z, False
    
def match(number, word, map1):
    map = map1.copy()

    for i in range(len(number)):
        if number[i] in map:
            if map[number[i]] != word[i]:
                return False
        else:
            if word[i] in map.values():
                return False
            map[number[i]] = word[i]
    return map

def sort_no_list(numbers):
    return sorted(numbers, key = lambda x: len(duplit_dict[decompose2(x)]) if decompose2(x) in duplit_dict else 0)
    

# %%
def find_letters_perfect(numbers, map, dict1):
    if len(numbers) == 0:
        return [map]
    encode = decompose2(numbers[0])
    maps = []
    if encode in dict1:
        for word in dict1[encode]:
            x = match(numbers[0], word, map)
            if x:
                maps += find_letters_perfect(numbers[1:], x, dict1)
        if len(maps) > 5:
            return maps[:5]
    return maps[:5]

def find_letters_imperfect(numbers, map, dict1):
    for i in range(len(numbers)):
        x = find_letters_perfect(numbers[i:], map, dict1)
        if len(x) > 0:
            return x[:5]
    return []


# %%
import random

def parse_(map, numbers, sorted):
    remainder = [chr(i) for i in range(65, 91) if chr(i) not in map.values()]
    letters = [i for i in ''.join(sorted) if i not in map.keys()]
    random.shuffle(remainder)
    px = remainder[:len(letters)]
    for i in range(len(letters)):
        map[letters[i]] = px[i]
    alpha = []
    for number in numbers:
        alpha.append(''.join([map[i] for i in number]))
    return alpha
        
def find_alphametic(numbers, dict1): 
    
    # Given a list of numbers return worded versions
    numbers = [str(i) for i in numbers]
    sorted = sort_no_list(numbers)
    maps, perfect = find_letters(sorted, {}, dict1)
    alphas = [parse_(map, numbers, sorted) for map in maps]
    return alphas, perfect
        

def find_alphametic_full(numbers):
    c_find, c_perfect = find_alphametic(numbers, common_word_dict)
    if c_perfect:
        return 'common', [(x, parse_alphametic(x)) for x in c_find]
    n_find, n_perfect = find_alphametic(numbers, normal_word_dict)
    if n_perfect:
        return 'normal', [(x, parse_alphametic(x)) for x in n_find]
    d_find, d_perfect = find_alphametic(numbers, duplit_dict)
    if d_perfect:
        return 'duplit', [(x, parse_alphametic(x)) for x in d_find]
    else:
        return 'failed', [(x, parse_alphametic(x)) for x in d_find] 

def parse_alphametic(words):
    return ' + '.join(words[:-1]) + ' = ' + words[-1]

# %%
def find_alphametic2(numbers, dict1, perfect = True): 
    
    # Given a list of numbers return worded versions
    numbers = [str(i) for i in numbers]
    sorted = sort_no_list(numbers)
    if perfect:
        maps = find_letters_perfect(sorted, {}, dict1)
    else:
        maps = find_letters_imperfect(sorted, {}, dict1)
    alphas = [parse_(map, numbers, sorted) for map in maps]
    return alphas
        

def find_alphametic_full2(numbers):
    c_find = find_alphametic2(numbers, common_word_dict)
    if c_find:
        return 'common', [(x, parse_alphametic(x)) for x in c_find]
    n_find = find_alphametic2(numbers, normal_word_dict)
    if n_find:
        return 'normal', [(x, parse_alphametic(x)) for x in n_find]
    d_find1 = find_alphametic2(numbers, duplit_dict)
    if d_find1:
        return 'duplit', [(x, parse_alphametic(x)) for x in d_find1]
    d_find2 = find_alphametic2(numbers, duplit_dict, perfect = False)
    if d_find2:
        return 'imperfect', [(x, parse_alphametic(x)) for x in d_find2]
    else:
        return 'failed'


# %%
cryptarithmetic_list = []
metics_list = list(categorize_alphametics(metics).values())
for group in metics_list:
    for numbers in group:
        x = find_alphametic2(numbers, common_word_dict)
        if x:
            cryptarithmetic_list.append({'Question': parse_alphametic(x[0]),
            'Answer': parse_alphametic([str(x) for x in numbers])})
            break
            
            
with open('3add3.txt', 'w') as f:
    for x in cryptarithmetic_list:
        json.dump(x, f)
        f.write('\n')



