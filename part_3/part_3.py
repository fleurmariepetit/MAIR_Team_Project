## part_3

import pandas as pd
import numpy as np
import re

words_and_types = pd.read_csv('wordtype_classification.csv')

maxIterations = 4
# Working
inputText = 'I\'m looking for Persian food please'.lower()
#inputText = 'Can I have an expensive restaurant'.lower()
#inputText = 'I\'m looking for world food'.lower()

#Almost working
#inputText = 'I want a restaurant that serves world food'.lower()
#inputText = 'I want a restaurant serving Swedish food'.lower()
#inputText = 'What is a cheap restaurant in the south part of town'.lower()
#inputText = 'Find a Cuban restaurant in the center'.lower()

# Not working
#inputText = 'I wanna find a cheap restaurant'.lower()
#inputText = 'What about Chinese food'.lower()
#inputText = 'I\'m looking for a moderately priced restaurant with Catalan food'.lower()
#inputText = 'I need a Cuban restaurant that is moderately priced'.lower()
#inputText = 'I\'m looking for an expensive restaurant and it should serve international food'.lower()
#inputText = 'I\'m looking for a restaurant in any area that serves Tuscan food'.lower()
#inputText = 'I\'m looking for a moderately priced restaurant in the west part of town'.lower()
#inputText = 'I would like a cheap restaurant in the west part of town'.lower()
#inputText = 'I\'m looking for a restaurant in the center'.lower()

inputText = re.sub(r'[^\w\s]', '', inputText).split()

print(inputText)
sentence_df = pd.DataFrame({'phrase': inputText, 'type1': np.nan, 'type2': np.nan, 'type3': np.nan},
                           columns=['phrase', 'type1', 'type2', 'type3'])

# Put data in data frame
for i in np.arange(len(inputText)):
    word = sentence_df['phrase'].iloc[i]
    types = words_and_types.loc[words_and_types['Word'] == word]

    if not types.empty:
        sentence_df['type1'].iloc[i] = types['Type1'].iloc[0]
        sentence_df['type2'].iloc[i] = types['Type2'].iloc[0]
        sentence_df['type3'].iloc[i] = types['Type3'].iloc[0]

    # todo: Check all the rows in which the 'type1' column is empty and use the python-Levenshtein library to map values to the closest domain term

# Reverse the order of the data frame
recent_Iteration = sentence_df.iloc[::-1]
print(recent_Iteration)

sentence_finished = False
iterations_list = []

# Add generation 0 to the data frame collection
recent_Iteration = pd.DataFrame(
    {'phrase': recent_Iteration['phrase'], 'type1': recent_Iteration['type1'], 'type2': recent_Iteration['type2'],
     'type3': recent_Iteration['type3'],
     'concatNumber': np.zeros(len(recent_Iteration['phrase']))},
    columns=['phrase', 'type1', 'type2', 'type3', 'concatNumber'])

iteration_number = 0
total_iteration = 1
iterations_list.append([0, recent_Iteration])
check_forward_slash = True

# Begin the tree algorithm
while not sentence_finished:
    iteration_DF = pd.DataFrame(
        {'phrase': recent_Iteration['phrase'], 'type1': recent_Iteration['type1'], 'type2': recent_Iteration['type2'],
         'type3': recent_Iteration['type3'],
         'concatNumber': recent_Iteration['concatNumber']},
        columns=['phrase', 'type1', 'type2', 'type3', 'concatNumber'])

    iteration_successful = False

    for i in np.arange(len(recent_Iteration)):
        current_type = recent_Iteration['type1'].iloc[i], recent_Iteration['type2'].iloc[i], \
                       recent_Iteration['type3'].iloc[i]

        if check_forward_slash:
            # Check if its not the last word, length instead of index because data frame starts at 0
            if i != (len(recent_Iteration) - 1):
                # TODO: Here I remove everything that is between brackets, this should be changed to remove everything from the FIRST brackets, otherwise errors can occur with types like '(np\s)/(np\s)' This is not that important though I gues
                # Checking request value of next word, where the next word is in fact the previous word in the sentence
                # TODO: Check for the multiple options that can get requested - The selected request type then has to be saved for a later use
                next_requested_type = re.sub(r'\([^)]*\)', '', recent_Iteration['type1'].iloc[i + 1])

                if '/' in next_requested_type:
                    next_requested_type = next_requested_type.split('/')[1]

                    if next_requested_type in current_type:
                        concatenation = recent_Iteration['phrase'].iloc[i + 1] + ' ' + recent_Iteration['phrase'].iloc[
                            i]
                        # Finding the new type
                        newType = recent_Iteration['type1'].iloc[i + 1]
                        newType = newType[:newType.rfind('/')]
                        # Removing parentheses
                        if newType.startswith('('):
                            newType = newType[1:-1]
                        # Replace the next word with the new information and remove current word. We also have to break the for loop because of index errors.
                        iteration_DF['phrase'].iloc[i + 1] = concatenation
                        iteration_DF['type1'].iloc[i + 1] = newType
                        iteration_DF['type2'].iloc[i + 1] = ''
                        iteration_DF['type3'].iloc[i + 1] = ''
                        iteration_DF['concatNumber'].iloc[i + 1] = iteration_DF['concatNumber'].iloc[i] + 1

                        iteration_DF = iteration_DF.drop(iteration_DF.index[i])
                        iteration_successful = True

                        iterations_list.append([iteration_number, iteration_DF])
                        break
        else:
            if i != (len(recent_Iteration) - 1):
                requested_type = re.sub(r'\([^)]*\)', '', recent_Iteration['type1'].iloc[i])

                if '\\' in requested_type:
                    requested_type = requested_type.split('\\')[0]

                    next_word_type = recent_Iteration['type1'].iloc[i + 1], recent_Iteration['type2'].iloc[i + 1], \
                                     recent_Iteration['type3'].iloc[i + 1]
                    if requested_type in next_word_type:
                        concatenation = recent_Iteration['phrase'].iloc[i + 1] + ' ' + recent_Iteration['phrase'].iloc[
                            i]
                        # Finding the new type
                        newType = recent_Iteration['type1'].iloc[i]
                        newType = newType[newType.rfind('\\')+1:]
                        # Removing parentheses
                        if newType.startswith('('):
                            newType = newType[1:-1]
                        # Replace the next word with the new information and remove current word. We also have to break the for loop because of index errors.
                        iteration_DF['phrase'].iloc[i] = concatenation
                        iteration_DF['type1'].iloc[i] = newType
                        iteration_DF['type2'].iloc[i] = ''
                        iteration_DF['type3'].iloc[i] = ''
                        iteration_DF['concatNumber'].iloc[i] = iteration_DF['concatNumber'].iloc[i] + 1

                        iteration_DF = iteration_DF.drop(iteration_DF.index[i + 1])
                        iteration_successful = True

                        iterations_list.append([iteration_number, iteration_DF])
                        break

    recent_Iteration = iteration_DF
    iteration_number += 1

    if iteration_successful == False:
        if check_forward_slash == False:
            if (total_iteration < maxIterations):
                check_forward_slash = True
                total_iteration += 1
            else:
                sentence_finished = True
        else:
            check_forward_slash = False

for iteration in iterations_list:
    print('~~~~~~~~~~~ Iteration '+str(iteration[0])+' ~~~~~~~~~~~~~')
    print(iteration[1])
