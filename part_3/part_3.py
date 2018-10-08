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
# s2 = empty_string
#
# For every letter (l) in input sentence (s):
#   to_lower_case(l)
#   if is_letter(l) or is_space(l)
#       s2 = concatenate(s2, l)
#
# words = string_to_list(seperate_str_on = " " )
# sentence_df = dataframe(words = words, types = NA)
#
# For every word in words:
#   get value of wat_df[,Type1] at word == wat_df[,Word]




