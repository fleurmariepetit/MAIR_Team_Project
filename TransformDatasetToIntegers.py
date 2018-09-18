import math
import random
import numpy as np

'''''
!!!!!!!!!!!!!!!!!!!!!
PLEASE READ : Every time you run this script the sentences are randomly shuffled, this will result in the fact that 
the words in the transformed sentences will be classified as different numbers than previous time.
!!!!!!!!!!!!!!!!!!!!!

General information
The sentences get transformed from words to a string of numbers.
Words are classified as '0' when the the word is not present in the library.
The number '1' is used as padding to make sure all the sentences contain the same amount of numbers.

Each sentence has a total of 23 numbers. This is based on the sentence that has the most amount of numbers in it

In the end it will generate two files, the first file is filled with sentences in which the words are now integers 
the second file contains the labels of these sentences (which category it is)
'''''

# Create the training and test set
def createTrainAndTestSets(completeDatasetFilename, trainingSetSize, trainingSetFilename,testSet):
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
    catCount = 0
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

# Transform the sentences to interger strings and
def transformSentences(wordDict,catDict, filename):
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

        return sentences,labels

completeDatasetFileName = 'acts_and_utts.txt'

trainingSetFileName = 'trainingSet.txt'
testSetFileName = 'testSet.txt'
trainingSetSizePercentile = 85

transformedTraingsSetFileName = 'transformedTrainingSet.txt'
transformedLabelsSetFileName = 'transformedLabelsSet.txt'

createTrainAndTestSets(completeDatasetFileName, trainingSetSizePercentile, trainingSetFileName, testSetFileName)

dictionaries = createWordAndCatogoryDictionaries('trainingSet.txt')
wordDict = dictionaries[0]
catDict = dictionaries[1]

transformResult = transformSentences(wordDict, catDict, 'trainingSet.txt')

sentences = transformResult[0]
labels = transformResult[1]

#Write sentences to text file
with open(transformedTraingsSetFileName, 'w') as target:
    for sentence in sentences:
        # This converts the array of numbers (words in the sentence) to a string separated by spaces
        target.write(" ".join(str(x) for x in sentence) + '\n')

#Write labels to text file
with open(transformedLabelsSetFileName, 'w') as target:
    for label in labels:
        # This converts the array of numbers (words in the sentence) to a string separated by spaces
        target.write(str(label) + '\n')

print("Done")