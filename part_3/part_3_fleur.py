## Fleur probeert dingen met de Levenshtein library

import pandas as pd
import numpy as np
import re
import Levenshtein as ls

words_and_types = pd.read_csv('C:/Users/Fleur/PycharmProjects/MAIR_Team_Project/part_3/wordtype_classification.csv')

inputText = 'im looking for world food troep im'.lower()
inputText = re.sub(r'[^\w\s]','',inputText).split()


print (inputText)
sentence_df = pd.DataFrame({'word': inputText, 'type1': np.nan, 'type2': np.nan, 'type3': np.nan},
                           columns=['word', 'type1', 'type2', 'type3'])

for i in np.arange(len(inputText)):
    word = sentence_df['word'].iloc[i]
    words_and_types["Distance"] = np.nan

    if not word in words_and_types["Word"].values:
        for j in np.arange(len(words_and_types)):
            Word = words_and_types["Word"].iloc[j]
            words_and_types["Distance"].iloc[j] = ls.distance(Word, word)

        min_dist = words_and_types["Distance"].min()
        types = words_and_types.loc[words_and_types["Distance"] == min_dist].iloc[0]
        word = types.iloc[0]
        sentence_df['word'].iloc[i] = word
    else:
        types = words_and_types.loc[words_and_types['Word'] == word].iloc[0]

    sentence_df['type1'].iloc[i] = types.iloc[1]
    sentence_df['type2'].iloc[i] = types.iloc[2]
    sentence_df['type3'].iloc[i] = types.iloc[3]

print(sentence_df)
