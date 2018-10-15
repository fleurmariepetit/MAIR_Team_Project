## part_3
import pandas as pd
import numpy as np
import math
import re
import Levenshtein as ls

words_and_types = pd.read_csv('wordtype_classification.csv')

# # Working
# inputText = 'I\'m looking for Persian food please'.lower()
# inputText = 'Can I have an expensive restaurant'.lower()
# inputText = 'I\'m looking for world food'.lower()
# inputText = 'What about Chinese food'.lower()
# inputText = 'What is a cheap restaurant in the south part of town'.lower()
# inputText = 'I want a restaurant serving Swedish food'.lower()
# inputText = 'I want a restaurant that serves world food'.lower()
# inputText = 'I wanna find a cheap restaurant'.lower()


# # Use different type-combination
#inputText = 'I\'m looking for a moderately priced restaurant with Catalan food'.lower()
#inputText = 'I need a Cuban restaurant that is moderately priced'.lower()
#inputText = 'I\'m looking for an expensive restaurant and it should serve international food'.lower()
#inputText = 'I\'m looking for a restaurant in any area that serves Tuscan food'.lower()
#inputText = 'I\'m looking for a moderately priced restaurant in the west part of town'.lower()
#inputText = 'I would like a cheap restaurant in the west part of town'.lower()
#inputText = 'I\'m looking for a restaurant in the center'.lower()

def prep_s_def(inputText):
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

    for k in range(0, len(sentence_df)):
        sentence_df.loc[k] = sentence_df.loc[k].fillna(sentence_df.loc[k, "type1"])

    sentence_df["type1"] = [re.sub(r'\/', 'FORW', x) for x in sentence_df["type1"]]
    sentence_df["type1"] = [re.sub(r'\\', 'BACK', x) for x in sentence_df["type1"]]
    sentence_df["type2"] = [re.sub(r'\/', 'FORW', x) for x in sentence_df["type2"]]
    sentence_df["type2"] = [re.sub(r'\\', 'BACK', x) for x in sentence_df["type2"]]
    sentence_df["type3"] = [re.sub(r'\/', 'FORW', x) for x in sentence_df["type3"]]
    sentence_df["type3"] = [re.sub(r'\\', 'BACK', x) for x in sentence_df["type3"]]
    return sentence_df

# TODO: Every iteration check if multiple types for a phrase. If so, and if possible constuct new type for each of the types.
# TODO: Then, save those alternative types for combined phrases in the columns type1, type2, and type3.
original_df = prep_s_def(inputText)
i = len(original_df) - 1
j = 0
second_try = False
tries = 0
type_tries = 0
max_tries = len(original_df) * 10
while (j < 3) & (not sentence_df["phrase"].str.contains(inputText).any()) & (tries < max_tries):
    del ccg_type, phrase, backward, forward, last_one, required_backward, prev_type, new_type, \
        required_forward, next_type, sentence_df
    j += 1
    type_tries = 0
    sentence_df = original_df[:]
    i = len(original_df) - 1
    print("-----------------------type_nr:", str(j), "--------------------------")
    while (not sentence_df["phrase"].str.contains(inputText).any()) & (type_tries < max_tries/3):
        type_tries += 1
        print("------------------------------" + str(i) + "----------------------------------")
        ccg_type = sentence_df.iloc[i, j]
        phrase = sentence_df.iloc[i, 0]
        backward = re.search(r"^((\w{1,2})|(\(\w{1,2}((FORW)|(BACK))\w{1,2}\)))(BACK)", ccg_type)
        forward = re.search(r"(FORW)(\w{1,2}$|(\(\w{1,2}((FORW)|(BACK))\w{1,2}\)$))", ccg_type)
        last_one = i
        if bool(backward):
            required_backward = re.search(r"^\w{1,2}|(^\(\w{1,2}((FORW)|(BACK))w{1,2}\))", ccg_type).group(0)
            required_backward = re.sub("[BACK]$", "", required_backward)
            if i != 0:
                prev_type = sentence_df.iloc[i - 1, j]
                if prev_type == required_backward:
                    new_type = re.sub(r"^((np)|(pp)|(s)|(n))(BACK)", '', ccg_type)
                    new_type = re.sub(r"\(|\)", "", new_type)
                    prev_phrase = sentence_df.iloc[i - 1, 0]
                    new_phrase = prev_phrase + " " + phrase
                    sentence_df.iloc[i, 0] = new_phrase
                    sentence_df.iloc[i, j] = new_type
                    sentence_df.drop(sentence_df.index[i - 1], inplace=True)
                    i -= 1
        elif bool(forward):
            required_forward = re.search(r"\w{1,2}$|\(\w{1,2}((FORW)|(BACK))\w{1,2}\)$", ccg_type).group(0)
            required_forward = re.sub("^[FORW]", "", required_forward)
            if bool(re.search(r"\)$", required_forward)):
                required_forward = re.sub(r"\)$", '', required_forward)
                required_forward = re.sub(r"^\(", '', required_forward)
            if i != (len(sentence_df)-1):
                next_type = sentence_df.iloc[i + 1, j]
                if next_type == required_forward:
                    new_type = re.sub(r"(FORW)((np)|(pp)|(s)|(n$))", '', ccg_type)
                    new_type = re.sub(r"(FORW)\(.*\)$", '', new_type)
                    if bool(re.search(r"\)$", new_type)):
                        new_type = re.sub(r"\)$", '', new_type)
                        new_type = re.sub(r"^\(", '', new_type)
                    next_phrase = sentence_df.iloc[i + 1, 0]
                    new_phrase = phrase + " " + next_phrase
                    sentence_df.iloc[i,0] = new_phrase
                    sentence_df.iloc[i, j] = new_type
                    sentence_df.drop(sentence_df.index[i + 1], inplace=True)
        if bool(re.search(r"(BACK)|(FORWARD)", sentence_df.iloc[i, j])) & (second_try == False):
            second_try = True
            print("Nog een keer met:" + str(i))
        elif (i < len(sentence_df)) & (i > 0):
            i -= 1
            second_try = False
            print("Een stap terug in de zin:" + str(i))
        elif (i==0) & (len(sentence_df) != 1):
            i += 1
        print(sentence_df)