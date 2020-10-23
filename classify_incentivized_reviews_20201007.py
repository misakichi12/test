'''
incentive review classification
'''

import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers

max_words = 1000
batch_size = 30
epochs = 5

data = pd.read_csv('data_train.csv')
data = data[['review_no','label', 'review_body']]
data = data.sample(10000, random_state=4222)
docs = data['review_body'].values.tolist()
num_classes = data['label'].nunique()

print("The number of classes is {}, and the number of observations is {}".format(num_classes, len(docs)))
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(docs)
x = tokenizer.texts_to_matrix(docs, mode='tfidf')
y = data['label'].values.tolist()
y = keras.utils.to_categorical(y, num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle =True, random_state=4222)
print("x_train shape {}, x_test shape {}, y_train shape {}, y_test shape {}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

print("Start to build the model...")
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), kernel_regularizer = regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

for i in range(0,1):
	model.add(Dense(512, kernel_regularizer = regularizers.l2(0.0001)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

model.add(Dense(num_classes,  kernel_regularizer = regularizers.l2(0.0001)))
model.add(Activation('sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs = epochs, verbose=1)

score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 1)

print("Test accuracy is {}".format(score[1]))