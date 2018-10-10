## part_3

import pandas as pd
import numpy as np
import re


words_and_types = pd.read_csv('wordtype_classification.csv')

inputText = 'im looking for world food troep'.lower()
inputText = re.sub(r'[^\w\s]','',inputText).split()


print (inputText)
sentence_df = pd.DataFrame({'word': inputText, 'type1': np.nan, 'type2': np.nan, 'type3': np.nan},
                           columns=['word', 'type1', 'type2', 'type3'])

for i in np.arange(len(inputText)):
    word = sentence_df['word'].iloc[i]
    types = words_and_types.loc[words_and_types['Word'] == word]

    if not types.empty:
        sentence_df['type1'].iloc[i] = types['Type1'].iloc[0]
        sentence_df['type2'].iloc[i] = types['Type2'].iloc[0]
        sentence_df['type3'].iloc[i] = types['Type3'].iloc[0]

    # todo: Check all the rows in which the 'type1' column is empty and use the python-Levenshtein library to map values to the closest domain term

print(sentence_df)


