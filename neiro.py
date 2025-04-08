import os
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.layers import *
from keras.utils import *

maxWordsCount = 1000
max_text_len = 50


def get_text(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        cont = f.readlines()
        cont[0] = cont[0].replace('\ufeff', '', 1)
    return cont


content = []
content.append(get_text('Общий класс.txt'))
content.append(get_text('Социально-экономический класс.txt'))
content.append(get_text('Физико-математический класс.txt'))
content.append(get_text('Химико-биологический класс.txt'))

texts = []
lens = []
for i in content:
    texts += i
    lens.append(len(i))
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–»—#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)


data = tokenizer.texts_to_sequences(texts)
x_train = pad_sequences(data, maxlen=max_text_len)

y_train = []
for i in range(len(lens)):
    y_train += [i] * lens[i]
y_train = to_categorical(y_train, len(lens))

indeces = np.random.choice(x_train.shape[0], size=x_train.shape[0], replace=False)
x_train = x_train[indeces]
y_train = y_train[indeces]

"""Building model"""

model = Sequential()
model.add(Embedding(input_dim=maxWordsCount, output_dim=128, input_length=max_text_len))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(rate=0.1))
model.add(LSTM(units=32))
model.add(Dense(len(lens), activation='softmax'))
model.summary()

save = model.to_json()
with open('model_configuration.json', 'w') as file:
    file.write(save)

""" Learning """
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=keras.optimizers.Adam(learning_rate=0.001))
history = model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=2, validation_split=0.2)
model.save_weights('model')

plt.plot(history.history['accuracy'], label='Точность распознавания направления на обучающей выборке', color='green')
plt.plot(history.history['val_accuracy'], label='Точность распознавания направления на проверочной выборке', color='red')
plt.legend()
plt.grid(color='red')
plt.show()


tokenizer = Tokenizer(lower=True)

with open('model_configuration.json', 'r') as file:
    model = model_from_json(file.read())
model.load_weights('model')

categories = list(map(lambda path: path[:-4].lower(), os.listdir('texts')))

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return words


t = input('Напишите о себе: ').lower()
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen = max_text_len)

res = model.predict(data_pad)
print('Ваше будушее направление {}!'.format(categories[np.argmax(res)]))
