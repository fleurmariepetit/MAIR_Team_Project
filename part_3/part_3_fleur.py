## part_3
import pandas as pd
import numpy as np
import math
import re
import Levenshtein as ls

words_and_types = pd.read_csv('wordtype_classification.csv')

maxIterations = 4
# # Working
inputText = 'I\'m looking for Persian food please'.lower()
# inputText = 'Can I have an expensive restaurant'.lower()
# inputText = 'I\'m looking for world food'.lower()
# inputText = 'What about Chinese food'.lower()
# inputText = 'I want a restaurant that serves world food'.lower()
# inputText = 'I want a restaurant serving Swedish food'.lower()

# # Almost working
# inputText = 'What is a cheap restaurant in the south part of town'.lower()
# inputText = 'Find a Cuban restaurant in the center'.lower()
# inputText = 'I wanna find a cheap restaurant'.lower()

# Not working
# inputText = 'I\'m looking for a moderately priced restaurant with Catalan food'.lower()
# inputText = 'I need a Cuban restaurant that is moderately priced'.lower()
# inputText = 'I\'m looking for an expensive restaurant and it should serve international food'.lower()
# inputText = 'I\'m looking for a restaurant in any area that serves Tuscan food'.lower()
# inputText = 'I\'m looking for a moderately priced restaurant in the west part of town'.lower()
# inputText = 'I would like a cheap restaurant in the west part of town'.lower()
# inputText = 'I\'m looking for a restaurant in the center'.lower()

inputText = re.sub(r'[^\w\s]', '', inputText)
input_list = inputText.split()

print(input_list)
sentence_df = pd.DataFrame({'phrase': input_list, 'type1': np.nan, 'type2': np.nan, 'type3': np.nan},
                           columns=['phrase', 'type1', 'type2', 'type3'])

# Put data in data frame
for i in np.arange(len(input_list)):
    word = sentence_df['phrase'].iloc[i]
    words_and_types["Distance"] = np.nan

    # Use Levenshtein distance to map mistyped words to the closest word in the vocabulary.
    if not word in words_and_types["Word"].values:
        for j in np.arange(len(words_and_types)):
            Word = words_and_types["Word"].iloc[j]
            words_and_types["Distance"].iloc[j] = ls.distance(Word, word)

        min_dist = words_and_types["Distance"].min()
        types = words_and_types.loc[words_and_types["Distance"] == min_dist].iloc[0]
        word = types.iloc[0]
        sentence_df['phrase'].iloc[i] = word
    else:
        types = words_and_types.loc[words_and_types['Word'] == word].iloc[0]

    sentence_df['type1'].iloc[i] = types.iloc[1]
    sentence_df['type2'].iloc[i] = types.iloc[2]
    sentence_df['type3'].iloc[i] = types.iloc[3]

print(sentence_df)

# TODO: Every iteration check if multiple types for a phrase. If so, and if possible constuct new type for each of the types.
# TODO: Then, save those alternative types for combined phrases in the columns type1, type2, and type3.

while (not sentence_df["phrase"].str.contains(inputText).any()):
    for i in range(len(sentence_df)-1, 0, -1):
        print("------------------------------" + str(i) + "----------------------------------")
        ccg_type = sentence_df.iloc[i,1]
        phrase = sentence_df.iloc[i,0]
        forward = re.search(r"[^\\]\w{1,2}$|\(\w{1,2}(\/|\\)\w{1,2}\)$", ccg_type)
        if not bool(forward):
            required_backward = re.search(r"^\w{1,2}|^\(\w{1,2}(\/|\\)w{1,2}\)", ccg_type).group(0)
            # Remove brackets
            required_backward = required_backward.replace("\(|\)", "")
            if i != 0:
                prev_type = sentence_df.iloc[i - 1,1]
                if prev_type == required_backward:
                    new_type = re.sub(r"^" + prev_type + r"\\", '', ccg_type)
                    new_type = re.sub(r"\(|\)", "", new_type)
                    prev_phrase = sentence_df.iloc[i - 1,0]
                    new_phrase = prev_phrase + " " + phrase
                    sentence_df.iloc[i] = new_phrase, new_type, "NaN", "NaN"
                    sentence_df.drop(sentence_df.index[i - 1], inplace=True)
                    print(sentence_df)
        elif bool(forward):
            required_forward = re.search(r"\w{1,2}$|\(\w{1,2}(\/|\\)\w{1,2}\)$", ccg_type).group(0)
            # Remove brackets
            required_forward = required_forward.replace("\(|\)", "")
            if i != (len(sentence_df)-1):
                next_type = sentence_df.iloc[i + 1,1]
                if next_type == required_forward:
                    new_type = re.sub(r"\/" + next_type + "$", '', ccg_type)
                    new_type = re.sub(r"\(|\)", "", new_type)
                    next_phrase = sentence_df.iloc[i + 1, 0]
                    new_phrase = phrase + " " + next_phrase
                    sentence_df.iloc[i] = new_phrase, new_type, "NaN", "NaN"
                    sentence_df.drop(sentence_df.index[i + 1], inplace=True)
                    print(sentence_df)