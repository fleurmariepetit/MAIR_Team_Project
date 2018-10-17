import pandas as pd
import queue
import re
import numpy as np


class Phrase:
    def __init__(self, text, type1, type2, type3, left_phrase, right_phrase, elimination_type):
        self.self = self
        self.text = text
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.left_phrase = left_phrase
        self.right_phrase = right_phrase
        self.elimination_type = elimination_type

    def forward_elimination(self, phrase):
        requested_types, original_types = phrase.get_forward_requests()

        if len(requested_types) > 0:
            new_types = []

            for i in np.arange(len(requested_types)):
                if requested_types[i] in self.get_types_array():
                    new_type = original_types[i]
                    new_type = new_type[:new_type.rfind('/')]
                    # Removing parentheses
                    if new_type.startswith('('):
                        new_type = new_type[1:-1]
                    new_types.append(new_type)
            if new_types:
                concatenation = phrase.text + ' ' + self.text
                first_type = new_types[0]
                second_type, third_type = np.nan, np.nan

                if len(new_types) > 1:
                    second_type = new_types[1]
                if len(new_types) > 2:
                    third_type = new_types[2]
                print(concatenation)

                return Phrase(text=concatenation, type1=first_type, type2=second_type, type3=third_type,
                              left_phrase=phrase,
                              right_phrase=self, elimination_type='Forward')
        return None

    def backward_elimination(self, phrase):
        requested_types, original_types = phrase.get_backward_requests()

        if len(requested_types) > 0:
            new_types = []

            for i in np.arange(len(requested_types)):
                if requested_types[i] in self.get_types_array():
                    # Finding the new type
                    new_type = original_types[i]
                    new_type = new_type[new_type.rfind('\\') + 1:]
                    # Removing parentheses
                    if new_type.startswith('('):
                        new_type = new_type[1:-1]
                    new_types.append(new_type)
            if new_types:
                concatenation = self.text + ' ' + phrase.text
                first_type = new_types[0]
                second_type, third_type = np.nan, np.nan

                if len(new_types) > 1:
                    second_type = new_types[1]
                if len(new_types) > 2:
                    third_type = new_types[2]
                print(concatenation)

                return Phrase(text=concatenation, type1=first_type, type2=second_type, type3=third_type,
                              left_phrase=self,
                              right_phrase=phrase, elimination_type='Backward')
        return None

    def get_forward_requests(self):
        request_types = [re.sub(r'^\([^)]*\)', '', self.type1)]
        if not pd.isnull(self.type2):
            request_types.append(re.sub(r'^\([^)]*\)', '', self.type2))
        if not pd.isnull(self.type3):
            request_types.append(re.sub(r'^\([^)]*\)', '', self.type3))

        return_types = []
        for requested_type in request_types:
            if '/' in requested_type:
                requested_type = requested_type.split('/')[1]
                requested_type = re.sub(r'^\(|\)$', '', requested_type)
                return_types.append(requested_type)

        return return_types, self.get_types_array()

    def get_backward_requests(self):
        request_types = [re.sub(r'^\([^)]*\)', '', self.type1)]
        if not pd.isnull(self.type2):
            request_types.append(re.sub(r'^\([^)]*\)', '', self.type2))
        if not pd.isnull(self.type3):
            request_types.append(re.sub(r'^\([^)]*\)', '', self.type3))

        return_types = []
        for requested_type in request_types:
            if '\\' in requested_type:
                requested_type = requested_type.split('\\')[0]
                requested_type = re.sub(r'^\(|\)$', '', requested_type)
                return_types.append(requested_type)
        return return_types, self.get_types_array()

    def get_types_array(self):
        return [self.type1, self.type2, self.type3]

    # This is the to_string function from the class (aka when you print the object)
    def __str__(self):
        return self.text


# Loading of stuff
words_and_types = pd.read_csv('wordtype_classification.csv')

inputText = 'I want a restaurant serving Swedish food'.lower()
inputText = re.sub(r'[^\w\s]', '', inputText).split()

startingSentence = []

# Translating the words with types to class instances
for word in inputText:
    # TODO error checking for non existing words
    types = words_and_types.loc[words_and_types['Word'] == word].iloc[0]
    startingSentence.append(
        Phrase(text=word, type1=types.iloc[1], type2=types.iloc[2], type3=types.iloc[3], left_phrase=None,
               right_phrase=None, elimination_type=None))

# Creating a queue. Items can put into the queue, when you .get() the queue you get an item by the FIFO(first in, first out) method. The item is then removed from the queue
sentence_queue = queue.Queue()
sentence_queue.put(startingSentence)

finished_trees = []
while not sentence_queue.empty():
    sentence_to_process = sentence_queue.get()

    # Check if sentence is fully processed. If this is the case the tree is done.
    if len(sentence_to_process) == 1:
        finished_trees.append(sentence_to_process)

    # Do forward elimination to all the words EXCEPT last word. When you get a successful elimination it will create a new sentence with the combined phrases and add it to the queue
    for i in np.arange(len(sentence_to_process) - 2):
        newPhrase = sentence_to_process[i + 1].forward_elimination(sentence_to_process[i])

        if newPhrase is not None:
            newSentence = sentence_to_process
            newSentence[i] = newPhrase
            del newSentence[i + 1]
            sentence_queue.put(newSentence)

    # Do backward elimination. TODO I'm not sure but it looks a bit like it's possible to merge the forward and elimination methods in the class.
    for i in np.arange(len(sentence_to_process) - 2):
        newPhrase = sentence_to_process[i].backward_elimination(sentence_to_process[i + 1])

        if newPhrase is not None:
            newSentence = sentence_to_process
            newSentence[i] = newPhrase
            del newSentence[i + 1]
            sentence_queue.put(newSentence)

# In the end the queue should be empty and all the possible trees should be in the finished_trees array.
# Printing the trees can be done by looping through the nested 'left_phrase' and 'right_phrase' from all the nodes until 'None' is found (this is set for the individual words)
# I think drawing the tree on a canvas could be fairly easy by using this nested structure as well but I have little expirience drawing in python so I'm not sure.
