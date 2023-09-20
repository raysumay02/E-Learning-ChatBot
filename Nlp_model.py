import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # Tokenize each word
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # add documents
        documents.append((wordList, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# lemmatize each word and remove duplicates and convert to Lowercase
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
# sort classes
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# create our training data
training = []
# create empty array for output
outputEmpty = [0] * len(classes)

# bag of words for training data
for document in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for pattern
    wordPatterns = document[0]
    # lemmatize each word - create base word, to represent related words
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    # create our bag of word if word match found
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1

    training.append(bag + outputRow)

# shuffle our features and turn into an nparray
random.shuffle(training)
training = np.array(training)

# create train and test lists X-patterns, Y-intents
trainX = training[:, :len(words)]
trainY = training[:, len(words):]


# Create the Neural network architecture with 3 layers
# First layer with 128 neurons, Second with 64 and Third with number of intents in the training data
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))


sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit and save the model
hist = model.fit(trainX, trainY, epochs=250, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('Done')
