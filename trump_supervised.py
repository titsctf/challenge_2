"""
Supervised Example
Example code to load the data and write a test submission, as well as evaluate 
a simple output using AUC. Output should be a file with one number per line, 
one for each corresponding test line, each denoting the probability of "FAKE".
"""
import csv
import numpy as np

with open('data/train2.csv', 'r') as f:
    reader = csv.reader(f)
    tweets_train = list(reader)
print ('train size', len(tweets_train))

with open('data/test2.csv', 'r') as f:
    reader = csv.reader(f)
    tweets_test2 = list(reader)
print ('test size', len(tweets_test2))

# Parse data
X = [t[0] for t in tweets_train]
y = [t[1]=='fake' for t in tweets_train]  # fake is 1, real is 0
X_test = [t[0] for t in tweets_test2]

# Split to train and val sets
X_train, X_val = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
y_train, y_val = y[:int(0.8*len(X))], y[int(0.8*len(X)):]
print ('train/val split:', len(X_train), len(X_val))
print ('ratio of positives', np.mean(y_train))

# Some example fake tweets
[tweet for (tweet, fake) in zip(X_train, y_train) if fake][:5]

# Some example real tweets
[tweet for (tweet, fake) in zip(X_train, y_train) if not fake][:5]


## Example: predict using a simple LSTM
########################################################################
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer

max_features = 20000
maxlen = 140  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

print('Parsing data...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train + X_val)
x_train = tokenizer.texts_to_sequences(X_train)
x_val = tokenizer.texts_to_sequences(X_val)
x_test = tokenizer.texts_to_sequences(X_test)
print(len(x_train), 'train sequences')
print(len(x_val), 'val sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('X_train shape:', x_train.shape)
print('X_val shape:', x_val.shape)
print('X_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_val, y_val))
score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)  # these are not that indicative because the data is unballanced...
########################################################################


y_score = model.predict(x_val)
from sklearn import metrics
print ('val AUC: ', metrics.roc_auc_score(y_val, y_score))
# this is a more sensible metric than accuracy: 0.66060839087675249


# Example writing output
y_score = model.predict(x_test)
with open("output2.csv", "w") as f:
    for y in y_score:
        f.write('%f\n' % y)
