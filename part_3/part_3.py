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

import pandas as pd
import sys
import re
import string

# Dataframe with words and types
wat_df = pd.read_csv("C:/Users/Fleur/PycharmProjects/MAIR_Team_Project/part_3/word-type.csv")

# user input
s = input("prompt")

s = s.lower()
s = re.sub(r'[^\w\s]','',s)

words = s.split()
sentence_df = pd.DataFrame(columns=['words', 'types'])

for word in words:
    type = wat_df.loc[wat_df["Word"] == word, "Type1"].iloc[0]
    sentence_row = pd.DataFrame([[word, type]], columns = ["words", "types"])
    sentence_df = sentence_df.append(sentence_row)

#-----------------------------------------------------------

### Example sentence_df:

## words   | types
##---------------
## im      | s
## looking | (s\s)/pp
## for     | pp/n
## world   | n/n
## food    | n

#-----------------------------------------------------------

## Combine types. This results in an extended sentence_df
# dataframe with atomic ánd combined types of the words in
# the input sentence.

# eventual_len = 2*len(sentences_df) - 1

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

#------------------------------------------------------------

### Example type-combination

# last_word = "looking for world food"
# previous_word = "im"
# combi = "im looking for world food"
# last_type = "s\s"
# previous_type = "s"
# type_combi = "s"
# sentences_df[9,] = ["im looking for world food", "s"]

#---------------------------------------------------------------



