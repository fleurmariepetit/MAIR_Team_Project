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
        # TODO do logic where you compare the phrase received with own types.
        # Try to do the forward elimination if successful return a new phrase
        # If not successful return None
        return Phrase(text='newText', type1='newType', type2='newType', type3='newtype', left_phrase=self,
                      right_phrase=phrase, elimination_type='Forward')

    def backward_elimination(self, phrase):
        # TODO Do the same as in forward elimination but than for the backslash
        return Phrase(text='newText', type1='newType', type2='newType', type3='newtype', left_phrase=self,
                      right_phrase=phrase, elimination_type='Backward')

    # This is the to_string function from the class (aka when you print the object)
    def __str__(self):
        return self.text


# Loading of stuff
words_and_types = pd.read_csv('wordtype_classification.csv')

inputText = 'I\'m looking for an expensive restaurant and it should serve international food'.lower()
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
    for i in np.arange(len(sentence_to_process) - 1):
        newPhrase = sentence_to_process[i].forward_elimination(sentence_to_process[i + 1])

        if newPhrase is not None:
            newSentence = sentence_to_process
            newSentence[i] = newPhrase
            del newSentence[i + 1]
            sentence_queue.put(newSentence)

    # Do backward elimination. TODO I'm not sure but it looks a bit like it's possible to merge the forward and elimination methods in the class. I'm too tired atm to be sure.
    for i in np.arange(len(sentence_to_process) - 1):
        newPhrase = sentence_to_process[i].backward_elimination(sentence_to_process[i + 1])

        if newPhrase is not None:
            newSentence = sentence_to_process
            newSentence[i] = newPhrase
            del newSentence[i + 1]
            sentence_queue.put(newSentence)


# In the end the queue should be empty and all the possible trees should be in the finished_trees array.
# Printing the trees can be done by looping through the nested 'left_phrase' and 'right_phrase' from all the nodes until 'None' is found (this is set for the individual words)
# I think drawing the tree on a canvas could be fairly easy by using this nested structure as well but I have little expirience drawing in python so I'm not sure.