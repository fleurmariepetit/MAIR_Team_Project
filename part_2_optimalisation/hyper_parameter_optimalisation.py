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
    return wordDict, catDict


# Transform the sentences to integer strings and
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

trainingSetFileName = 'trainingSet.txt'
testSetFileName = 'testSet.txt'
trainingSetSizePercentile = 85

transformedTraingsSetFileName = 'transformedTrainingSet.txt'
transformedLabelsSetFileName = 'transformedLabelsSet.txt'

createTrainAndTestSets(completeDatasetFileName, trainingSetSizePercentile, trainingSetFileName, testSetFileName)

wordDict, catDict = createWordAndCatogoryDictionaries('trainingSet.txt')

transformResult = transformSentences(wordDict, catDict, 'trainingSet.txt')
transformTestset = transformSentences(wordDict, catDict, 'testSet.txt')

trainData = np.array(transformResult[0])
trainLabels = transformLabels(transformResult[1])

testData = np.array(transformTestset[0])
testLabels = transformLabels(transformTestset[1])

with open('part_2_optimalisation/catDict.json', 'w') as cd:
    json.dump(catDict, cd, sort_keys=True, indent=4)

with open('part_2_optimalisation/wordDict.json', 'w') as wd:
    json.dump(wordDict, wd, sort_keys=True, indent=4)

# Optimalisation
with open('part_2_optimalisation/catDict.json', 'r') as cd:
    catDict = json.load(cd)
with open('part_2_optimalisation/wordDict.json', 'r') as wd:
    wordDict = json.load(wd)

accuracies = []
losses = []
plot_it = False

def fit_lstm(emb_out, lstm_un, lstm_in, dense_in, nr_epoch):
    model = Sequential()
    model.add(Embedding(input_dim=len(wordDict) + 2,
                        output_dim=emb_out,
                        input_length=23,
                        mask_zero=True))
    model.add(LSTM(units=lstm_un,
                   activation='relu',
                   input_shape=(lstm_in,)))
    model.add(Dense(units=len(catDict),
                    activation='softmax',
                    input_shape=(dense_in,)))
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(trainData,
                        trainLabels,
                        epochs=nr_epoch,
                        validation_data=(testData, testLabels))

    accuracies.append(history.history['acc'][-1])
    losses.append(history.history['loss'][-1])

    # Adapted from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    # Visited: 01-10-2018
    if plot_it:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss and accuracy')
        plt.ylabel('loss/accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
        plt.savefig('part_2_optimalisation/accloss_{0}-{1}-{2}-{3}-{4}.png'.format(emb_out, lstm_un, lstm_in, dense_in, nr_epoch))
        model.save('part_2_optimalisation/model{0}eps.h5'.format(nr_epoch))

        # summarize history for loss
 #       plt.plot(history.history['loss'])
 #       plt.plot(history.history['val_loss'])
 #       plt.title('model loss')
 #       plt.ylabel('loss/accuracy')
 #       plt.xlabel('epoch')
 #       plt.legend(['train', 'test'], loc='upper left')
 #       plt.savefig('loss{0}{1}{2}{3}{4}.png'.format(emb_out, lstm_un, lstm_in, dense_in, nr_epoch))


# Make the list with all the different parameter options. In this case a vary between different combinations
# of 128, 64, and 32 neurons. They have been tested on 20 epochs. The optimum combination for this test run was
# emb_out = 128, lstm_un = 64, lstm_in = 128, dense_in = 32. We looked at the graphs of the best configuration
# for 20 epochs and decided that 7 epochs would be enough. We tested all different combinations again for just 7 epochs.
# The optimal combination of our test round was emb_out = 128, lstm_un = 64, lstm_in = 128, dense_in = 32.
# The worst accuracy was for 32, 64, 64, 32, the highest loss for 32, 64, 64, 32, 7. Optimal settings reach an accuracy
# of .9702 on test set, and .9803 on trainingsset.
neurons = []
nr_epochs = [7]
for i in range(5, 8):
    neurons.append(2 ** i)
hyper_pars = [neurons, neurons, neurons, neurons, nr_epochs]
fit_args = list(itertools.product(*hyper_pars))
max_acc_pars = []
min_loss_pars = []
min_acc = int()
max_acc = int()
min_loss = int()
max_loss = int()


# This function is used to check what would be the optimal parameter settings for
# the number of neurons. All possible combinations of 128, 64, and 32 where tested.
# The hyper paremeter setting that scored best where 128, 128, 32, 64 at 20 epochs.
def opt_neurons():
    for j in range(0, len(fit_args)):
        emb_out = fit_args[j][0]
        lstm_un = fit_args[j][1]
        lstm_in = fit_args[j][2]
        dense_in = fit_args[j][3]
        nr_epoch = fit_args[j][4]
        fit_lstm(emb_out, lstm_un, lstm_in, dense_in, nr_epoch)


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

    max_acc = max(accuracies)
    print(max_acc)
    max_acc_pars = fit_args[accuracies.index(max_acc)]
    print(max_acc_pars)

    # The hyper parameters with the least loss
    min_loss = min(losses)
    print(min_loss)
    min_loss_pars = fit_args[losses.index(min_loss)]
    print(min_loss_pars)

    # The worst accuracy
    min_acc = min(accuracies)
    print(min_acc)
    min_acc_pars = fit_args[accuracies.index(min_acc)]
    print(min_acc_pars)

    # The highest loss
    max_loss = max(losses)
    print(max_loss)
    max_loss_pars = fit_args[losses.index(max(losses))]
    print(max_loss_pars)


# Tweak number of epochs on basis of plots, keeping the hyper parameters the same.
def opt_epochs():
    get_optima()
    emb_out = max_acc_pars[0]
    lstm_un = max_acc_pars[1]
    lstm_in = max_acc_pars[2]
    dense_in = max_acc_pars[3]
    nr_epoch = 7
    global plot_it
    plot_it = True
    fit_lstm(emb_out, lstm_un, lstm_in, dense_in, nr_epoch)