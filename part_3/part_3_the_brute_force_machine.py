import pandas as pd
import queue
import re
import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx


class Phrase:
    def __init__(self, text, type1, type2, type3, left_phrase, right_phrase, elimination_type, build_history):
        self.self = self
        self.text = text
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.left_phrase = left_phrase
        self.right_phrase = right_phrase
        self.elimination_type = elimination_type
        self.build_history = build_history

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

                build_history = phrase.text + '(' + str(phrase.get_types_array()) + ') + ' + self.text + '(' + str(
                    self.get_types_array()) + ')' + ' created: "' + phrase.text + ' ' + self.text + '" (' + str(
                    new_types) + ')'
                return Phrase(text=concatenation, type1=first_type, type2=second_type, type3=third_type,
                              left_phrase=phrase,
                              right_phrase=self, elimination_type='/E', build_history=build_history)
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

                build_history = self.text + '(' + str(self.get_types_array()) + ') + ' + phrase.text + '(' + str(
                    phrase.get_types_array()) + ')' + ' created: "' + self.text + ' ' + phrase.text + '" (' + str(
                    new_types) + ')'
                return Phrase(text=concatenation, type1=first_type, type2=second_type, type3=third_type,
                              left_phrase=self,
                              right_phrase=phrase, elimination_type='\\E', build_history=build_history)
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

    # To check if the tree (or part of the tree) contains a leaf consisting of the key value pair as leafs
    def contains_key_value_leaf_connection(self, key, value):
        if str(self.left_phrase) != 'None' and str(self.right_phrase) != 'None':
            if (self.left_phrase.text == key and self.right_phrase.text == value) or (
                    self.left_phrase.text == value and self.right_phrase.text == key):
                return True
            elif self.left_phrase.contains_key_value_leaf_connection(key, value):
                return True
            elif self.right_phrase.contains_key_value_leaf_connection(key, value):
                return True
        return False

    # This is the to_string function from the class (aka when you print the object)
    def __str__(self):
        return self.text


# Recursive method to get a textual representation of the nodes in the tree and how they are connected
def traverse(phrase):
    if (phrase.build_history != ''):
        print(phrase.build_history)
    if (str(phrase.left_phrase) != 'None'):
        edges.append((str(phrase), str(phrase.left_phrase)))
        labels[(str(phrase), str(phrase.left_phrase))] = ''
        traverse(phrase.left_phrase)
    if (str(phrase.right_phrase) != 'None'):
        edges.append((str(phrase), str(phrase.right_phrase)))
        labels[(str(phrase), str(phrase.right_phrase))] = ''
        traverse(phrase.right_phrase)
    return


# https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
# Needed for drawing of the canvas in the end
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1 / levels[currentLevel][TOTAL]
        left = dx / 2
        pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
        return pos

    if levels is None:
        levels = make_levels({})
    else:
        levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})


def draw_image(edges):
    # Drawing of the end result on the canvas
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = hierarchy_pos(G, edges[0][0])
    nx.draw(G, pos=pos, with_labels=True, node_color='w')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.2)
    plt.show()


# Loading of data
words_and_types = pd.read_csv('wordtype_classification.csv')
food_types = pd.read_csv('food_type.csv')
price_types = pd.read_csv('price_type.csv')
location_types = pd.read_csv('location_type.csv')

# Working phrases
# inputText = 'I\'m looking for Persian food please'.lower()
# inputText = 'Can I have an expensive restaurant'.lower()
# inputText = 'I\'m looking for world food'.lower()
# inputText = 'What about Chinese food'.lower()
# inputText = 'I want a restaurant serving Swedish food'.lower()
# inputText = 'I want a restaurant that serves world food'.lower()
# inputText = 'I need a Cuban restaurant that is moderately priced'.lower()
#inputText = 'I wanna find a cheap restaurant'.lower()
#inputText = 'What is a cheap restaurant in the south part of town'.lower()
#inputText = 'I\'m looking for a moderately priced restaurant with Catalan food'.lower()
# inputText = 'I\'m looking for a restaurant in any area that serves Tuscan food'.lower()
# inputText = 'I\'m looking for a restaurant in the center'.lower()
inputText = 'Find a Cuban restaurant in the center'.lower()
#inputText = 'I\'m looking for an expensive restaurant and it should serve international food'.lower()

# Not working phrases
# inputText = 'I\'m looking for a moderately priced restaurant in the west part of town'.lower()
# inputText = 'I would like a cheap restaurant in the west part of town'.lower()

inputText = re.sub(r'[^\w\s]', '', inputText).split()

startingSentence = []

# Translating the words with types to class instances
for word in inputText:
    # TODO error checking for non existing words and implement Levenshtein
    types = words_and_types.loc[words_and_types['Word'] == word].iloc[0]
    startingSentence.append(
        Phrase(text=word, type1=types.iloc[1], type2=types.iloc[2], type3=types.iloc[3], left_phrase=None,
               right_phrase=None, elimination_type=None, build_history=''))

# Items can put into the queue, when you .get() the queue you get an item by the FIFO method (first in, first out). The item is then removed from the queue
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
        newPhrase = sentence_to_process[i + 1].forward_elimination(sentence_to_process[i])
        if newPhrase is not None:
            newSentence = copy.deepcopy(sentence_to_process)
            newSentence[i] = newPhrase
            del newSentence[i + 1]
            sentence_queue.put(newSentence)

            ''''' This code can be used to see the steps the system makes, useful for debugging sentences that won't complete. Don't forget to put this code in the backward elimination as well
            s = ''
            for sent in newSentence:
                s = s + ',' + sent.text
            print('sentence: ' + s)
            '''''

    # Do backward elimination.
    for i in np.arange(len(sentence_to_process)):
        newPhrase = sentence_to_process[i - 1].backward_elimination(sentence_to_process[i])

        if newPhrase is not None:
            newSentence = copy.deepcopy(sentence_to_process)
            newSentence[i] = newPhrase
            del newSentence[i - 1]
            sentence_queue.put(newSentence)

print('Amount of trees generated:' + str(len(finished_trees)))
finished_trees = [item[0] for item in finished_trees]
labels = {}
edges = []

# Types of user preferences that are present in some form in the user input
food_preference = list(set(inputText) & set(np.concatenate(food_types.values.tolist(), axis=0)))
price_preference = list(set(inputText) & set(np.concatenate(price_types.values.tolist(), axis=0)))
location_preference = list(set(inputText) & set(np.concatenate(location_types.values.tolist(), axis=0)))

possible_trees = []

for tree in finished_trees:
    add_tree = True
    if len(food_preference) > 0:
        add_tree = (tree.contains_key_value_leaf_connection(food_preference[0], 'food')) or (
            tree.contains_key_value_leaf_connection(food_preference[0], 'restaurant'))
    if (len(price_preference) > 0) and (add_tree is True):
        add_tree = (tree.contains_key_value_leaf_connection(price_preference[0], 'priced')) or (
            tree.contains_key_value_leaf_connection(price_preference[0], 'restaurant'))
    if (len(location_preference) > 0) and (add_tree is True):
        add_tree = (tree.contains_key_value_leaf_connection(location_preference[0], 'part')) or (tree.contains_key_value_leaf_connection(location_preference[0], 'the'))

    if (add_tree):
        possible_trees.append(tree)

print('Amount of trees with correct preference:' + str(len(possible_trees)))

if (len(possible_trees) > 0):
    traverse(possible_trees[0])

    preferences = ['disjoint / not present', 'disjoint / not present', 'disjoint / not present']
    if len(food_preference) > 0:
        preferences[0] = food_preference[0]
    if (len(price_preference) > 0):
        preferences[1] = price_preference[0]
    if (len(location_preference) > 0):
        preferences[2] = location_preference[0]

    print('Food preference :' + preferences[0])
    print('Price range: ' + preferences[1])
    print('Location preference: ' + preferences[2])
else:
    traverse(finished_trees[0])
    print('Food preference: disjoint / not present')
    print('Price range: disjoint / not present')
    print('Location preference: disjoint / not present')

draw_image(edges)
