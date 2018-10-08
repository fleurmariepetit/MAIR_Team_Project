## part_3

# ---------------------------------
# Example sentences

# 1. im looking for world food
# S -> S S\S
# S\S -> (S\S)/PP PP
# PP -> PP/N N
# N -> N/N N
# S -> im
# (S\S)/PP -> looking
# PP/N -> for
# N/N -> world
# N -> food

# 2. i want a restaurant that serves world food
# S -> S/NP NP
# S/NP -> NP NP\(S/NP)
# NP -> NP NP\NP
# NP -> NP/N N
# NP\NP -> (NP\NP)/N N
# N -> N/N N
# (NP\NP)/N -> ((NP\NP)/N)/(NP\S) NP\S
# NP -> i
# NP\(S/NP) -> want
# NP/N -> a
# N -> restaurant
# ((NP\NP)/N)/(NP\S) -> that
# NP/S -> serves
# N/N -> world
# N -> food

# -----------------------------------
# Plan in pseudocode:
#
# words_and_types = word-type.csv
# wat_df = to_dataframe(words_and_types)
# s = input
# s1 = ""
#
# For every l in input s:
#   to_lower_case(l)
#   if is_letter(l) or is_space(l)
#       s1 = concatenate(s1, l)
#
# words = string_to_list(s1, seperate_str_on = " " )
# sentence_df = dataframe()
#
# For every word in words:
#   type = value of wat_df[,Type1] at word == wat_df[,Word]
#   sentence_df = dataframe(words = word, types = type)
#
# -----------------------------------------------------------
#

import pandas as pd
import numpy as np

words_and_types = pd.read_csv('wordtype_classification.csv')

inputText = 'im looking for world food'.lower().split()

sentence_df = pd.DataFrame({'word': inputText, 'type1': np.nan, 'type2': np.nan, 'type3': np.nan},
                           columns=['word', 'type1', 'type2', 'type3'])

for i in np.arange(len(inputText)):
    word = sentence_df['word'].iloc[i]
    types = words_and_types.loc[words_and_types['Word'] == word]

    sentence_df['type1'].iloc[i] = types['Type1'].iloc[0]
    sentence_df['type2'].iloc[i] = types['Type2'].iloc[0]
    sentence_df['type3'].iloc[i] = types['Type3'].iloc[0]

print(sentence_df)


### Example sentence_df:
## words   | types
##---------------
## im      | s
## looking | (s\s)/pp
## for     | pp/n
## world   | n/n
## food    | n
#
# -----------------------------------------------------------
#
## Combine types. This results in an extended senetence_df
# dataframe with atomic and combined types of the words in
# the input sentence.
#
# eventual_len = 2*len(sentences_df) - 1
#
# for i in range(1, eventual_len, 2):
#   last_word = sentences_df[len(sentences_df), words].
#   previous_word = sentences_df[len(sentences_df) - i, words]
#   combi = concatenate(previous_word, last_word)
#   last_type = sentences_df[len(sentences_df), types]
#   previous_type = sentences_df[len(sentences_df) - i, types]
#   if str_match(previous_type, reg_ex = concatenate("/", last_type, "$"))
#       type_combi = str_remove(previous_type, regex = concatenate("/", last_type, ")?$"))
#   else
#       type_combi = str_remove(last_type, regex = concatenate("^(?", previous_type, "\"))
#   type_combi = str_remove(type_combi, regex = ")|(")
#   sentences_df[len(sentences_df) + 1,] = [combi, type_combi]
#
# ------------------------------------------------------------
#
### Example type-combination
#
# last_word = "looking for world food"
# previous_word = "im"
# combi = "im looking for world food"
# last_type = "s\s"
# previous_type = "s"
# type_combi = "s"
# sentences_df[9,] = ["im looking for world food", "s"]
#
