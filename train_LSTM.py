# Vocabulary size = 784
# Sequence length = 23
# Length trainingset = 25501

# The following code is adapted from: https://www.youtube.com/watch?v=iMIWee_PXl8
# Visited on 19-09-2018

from keras.layers import Input, Embedding, LSTM, Dense, Flatten
from keras.models import Model, Sequential
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Transform training file lines to arrays
training_set = open('transformed_TnT.txt', 'r')

split_lines = [line.split(' ') for line in training_set.readlines()]

for i in range(len(split_lines)):
    line = split_lines[i]
    for j in range(len(line)):
        #int_arr = []
        ch_int = int(line[j])
        #int_arr.append(ch_int)
        #line[j] = int_arr
        line[j] = ch_int

# Print vocabulary size.
print(np.amax(split_lines))

data = np.array(split_lines, dtype=float)
print(data.shape)

# Transform labels to array
labels_arr = []
labels_set = open('transformed_TnT_labels.txt', 'r')
for line in labels_set:
    labels_arr.append(int(line))

print(labels_arr)
target = np.array(labels_arr, dtype = float)
print(target.shape)

# Split test and train data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size= .15)

# Define model
model = Sequential()
model.add(Embedding(785, 1, input_length = 23, mask_zero = True))
model.add(LSTM((1), batch_input_shape = (None,23,1),return_sequences=True, activation='softmax', recurrent_activation = 'relu'))
#model.add(LSTM((1), return_sequences = False))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

# Train it. 2 epochs just for testing. Training more epochs takes longer but
# should increase accuracy.
history = model.fit(x_train, y_train, epochs = 2, validation_data = (x_test, y_test))

plt.plot(history.history['loss'])
plt.show()
