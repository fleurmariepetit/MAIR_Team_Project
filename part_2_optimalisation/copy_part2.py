# Optimalisation of hyper parameters

import math
import random
import matplotlib.pyplot as plt
import itertools

import numpy as np
import json
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential, load_model


# Create the training and test set
def createTrainAndTestSets(completeDatasetFilename, trainingSetSize, trainingSetFilename, testSet):
    with open(completeDatasetFilename, 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()

    trainingSize = math.floor(len(data) / 100 * trainingSetSize)

    with open(trainingSetFilename, 'w') as target:
        for i in range(0, trainingSize):
            target.write(data[i][1])

    with open(testSet, 'w') as target:
        for i in range(trainingSize, len(data)):
            target.write(data[i][1]);


# Create the word and category dictionaries
def createWordAndCatogoryDictionaries(fileName):
    wordcount = 2
    catCount = 1
    lineCount = 0
    wordDict = dict()  # Word dictionary
    catDict = dict()  # Category dictionary

    with open(fileName, 'r') as f:
        for line in f:
            lineCount += 1
            newline = True

            for word in line.split():
                if newline:
                    newline = False
                    if word not in catDict:
                        catDict[word] = catCount
                        catCount += 1
                else:
                    if word not in wordDict:
                        wordDict[word] = wordcount
                        wordcount += 1

        with open('part_2_optimalisation/catDict7.json', 'w') as cd:
            json.dump(catDict, cd, sort_keys=True, indent=4)

        with open('part_2_optimalisation/wordDict7.json', 'w') as wd:
            json.dump(wordDict, wd, sort_keys=True, indent=4)

    return wordDict, catDict


#Transform the sentences to integer strings and
with open('part_2_optimalisation/catDict.json', 'r') as cd:
    catDict = json.load(cd)
with open('part_2_optimalisation/wordDict.json', 'r') as wd:
    wordDict = json.load(wd)

def transformSentences(wordDict, catDict, filename):
    with open(filename, 'r') as f:
        sentences = []
        labels = []

        for line in f:
            sentence = np.ones((23,), dtype=int)
            newline = True
            wordCount = 0

            for word in line.split():
                if newline:
                    labels.append(catDict[word])
                    newline = False
                else:
                    if word in wordDict:
                        sentence[wordCount] = wordDict[word]
                    else:
                        sentence[wordCount] = 0
                    wordCount += 1

            sentences.append(sentence)

        return sentences, labels


def transformUserSentence(wordDict, sentence):
    transformedSentence = np.ones((23,), dtype=int)
    wordCount = 0

    for word in sentence.split():
        if word in wordDict:
            transformedSentence[wordCount] = wordDict[word]
        else:
            transformedSentence[wordCount] = 0
        wordCount += 1
        if wordCount > 23:
            break

    return transformedSentence


def transformLabels(labels):
    newTrainLabels = []
    for row in labels:
        newRow = np.zeros(15, dtype=int)
        if row is not 0:
            newRow[row - 1] = 1
        newTrainLabels.append(newRow)
    return np.array(newTrainLabels)


completeDatasetFileName = 'part_2_optimalisation/acts_and_utts.txt'

trainingSetFileName = 'part_2_optimalisation/trainingSet.txt'
testSetFileName = 'part_2_optimalisation/testSet.txt'
trainingSetSizePercentile = 85

transformedTraingsSetFileName = 'part_2_optimalisation/transformedTrainingSet.txt'
transformedLabelsSetFileName = 'part_2_optimalisation/transformedLabelsSet.txt'

createTrainAndTestSets(completeDatasetFileName, trainingSetSizePercentile, trainingSetFileName, testSetFileName)

wordDict, catDict = createWordAndCatogoryDictionaries('part_2_optimalisation/trainingSet.txt')

transformResult = transformSentences(wordDict, catDict, 'part_2_optimalisation/trainingSet.txt')
transformTestset = transformSentences(wordDict, catDict, 'part_2_optimalisation/testSet.txt')

trainData = np.array(transformResult[0])
trainLabels = transformLabels(transformResult[1])

testData = np.array(transformTestset[0])
testLabels = transformLabels(transformTestset[1])

# Optimalisation
accuracies = []
losses = []

def fit_lstm(emb_out, lstm_un, nr_epoch):
    model = Sequential()
    model.add(Embedding(input_dim=len(wordDict) + 2,
                        output_dim=emb_out,
                        input_length=23,
                        mask_zero=True))
    model.add(LSTM(units=lstm_un,
                   activation='relu'))
    model.add(Dense(units=len(catDict),
                    activation='softmax'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(trainData,
                        trainLabels,
                        epochs=nr_epoch,
                        validation_data=(testData, testLabels))

    accuracies.append(history.history['acc'])
    losses.append(history.history['loss'])

    # Adapted from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    # Visited: 01-10-2018
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss and accuracy')
    plt.ylabel('loss/accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
    plt.savefig('part_2_optimalisation/accloss_{0}-{1}-{2}.png'.format(emb_out, lstm_un, nr_epoch))
    plt.clf()

    model.save('part_2_optimalisation/model{0}eps.h5'.format(nr_epoch))

        # summarize history for loss
 #       plt.plot(history.history['loss'])
 #       plt.plot(history.history['val_loss'])
 #       plt.title('model loss')
 #       plt.ylabel('loss/accuracy')
 #       plt.xlabel('epoch')
 #       plt.legend(['train', 'test'], loc='upper left')
 #       plt.savefig('loss{0}{1}{2}{3}{4}.png'.format(emb_out, lstm_un, lstm_in, dense_in, nr_epoch))


# Best with 20 epochs scored 8s 703us/step - loss: 0.0292 - acc: 0.9919 - val_loss: 0.0820 - val_acc: 0.9846.
# The settings where emb_out = 128, lstm_un = 32. Seems to be stable after 8 epochs
# Let's look find the best settings after 8 epochs.
neurons = []
nr_epochs = [20]
for i in range(5, 8):
    neurons.append(2 ** i)
hyper_pars = [neurons, neurons, nr_epochs]
fit_args = list(itertools.product(*hyper_pars))
max_acc_pars = []
min_loss_pars = []
min_acc = int()
max_acc = int()
min_loss = int()
max_loss = int()


# This function is used to check what would be the optimal parameter settings for
# highest accuracy:
# 0.9954640297559648
# highest accuracy parameters:
# [128, 64, 20]
# smallest loss:
# 0.020156365977649776
# smallest loss parameters:
# [128, 64, 20]
# highest loss:
# 0.04916552006152267
# highest loss parameters:
# [32, 32, 1]

# 10 epochs:
# 0.9901115848680032
# [128, 64, 10]

# 10 epoch model:
# loss: 0.0411 - acc: 0.9875 - val_loss: 0.1028 - val_acc: 0.9805
def opt_neurons():
    for j in range(0, len(fit_args)):
        emb_out = fit_args[j][0]
        lstm_un = fit_args[j][1]
        nr_epoch = fit_args[j][2]
        fit_lstm(emb_out, lstm_un, nr_epoch)


def get_optima():
    # The hyper parameter with the highest accuracy
    global max_acc
    global max_acc_pars
    global max_loss
    global max_loss_pars
    global min_acc
    global min_acc_pars
    global min_loss
    global min_loss_pars

    max_accs = []
    max_ainds = []

    min_losss = []
    min_linds = []

    min_accs = []
    min_ainds = []

    max_losss = []
    max_linds = []


    # The parameters with the highest accuracy
    for i in range(len(accuracies) - 1):
        max_accs.append(max(accuracies[i]))
        max_ainds.append(accuracies[i].index(max(accuracies[i])))

    max_acc = max(max_accs)
    print("highest accuracy: ")
    print(max_acc)

    # tuple: (index of max acc, nr of epochs)
    max_aind = [max_accs.index(max_acc), max_ainds[max_accs.index(max_acc)]]

    max_acc_pars = list(fit_args[max_aind[0]])
    max_acc_pars[2] = max_aind[1] + 1
    print("highest accuracy parameters:")
    print(max_acc_pars)

    # The parameters with the least loss
    for i in range(len(losses) - 1):
        min_losss.append(min(losses[i]))
        min_linds.append(losses[i].index(min(losses[i])))

    min_loss = min(min_losss)
    print("smallest loss: ")
    print(min_loss)

    min_lind = [min_losss.index(min_loss), min_linds[min_losss.index(min_loss)]]

    min_loss_pars = list(fit_args[min_lind[0]])
    min_loss_pars[2] = min_lind[1] + 1
    print("smallest loss parameters: ")
    print(min_loss_pars)

    # The highest loss
    for i in range(len(losses) - 1):
        max_losss.append(min(losses[i]))
        max_linds.append(losses[i].index(max(losses[i])))

    max_loss = max(max_losss)
    print("highest loss: ")
    print(max_loss)

    max_lind = [max_losss.index(max_loss), max_linds[max_losss.index(max_loss)]]

    max_loss_pars = list(fit_args[max_lind[0]])
    max_loss_pars[2] = max_lind[1] + 1
    print("highest loss parameters: ")
    print(max_loss_pars)


# Best of n epochs
def best_ofn_epochs():
    n = 10
    accs_n = []

    for i in range(len(accuracies) - 1):
        accs_n.append(accuracies[i][n])

    maccs_n = max(accs_n)
    print(maccs_n)

    maccs_n_ind = accs_n.index(maccs_n)
    maccs_n_args = list(fit_args[maccs_n_ind])
    maccs_n_args[2] = n