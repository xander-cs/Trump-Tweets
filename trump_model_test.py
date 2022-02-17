import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, TimeDistributed
from keras.models import load_model, model_from_json
import pickle

print("\nImporting Dataset...\n")

df = pd.read_csv("realdonaldtrumpedited.csv",parse_dates=True)  
df.text = df.text.str.lower()
df.date = pd.to_datetime(df.date)
df = df.set_index("date")
df.text = df.text.str.replace(r"http[\w:/\.]+","") # remove urls
df.text = df.text.str.replace(r"pic.twitter.com+","")
df.text = df.text.str.replace("www.youtube.com/user/mattressserta","")
df.text = df.text.str.replace(r'[^!\'"#$%&\()*+,-./:;<=>?@_’`{|}~\w\s]',' ') #remove everything but characters and punctuation
df.text = df.text.str.replace("  "," ") #replace multple white space with a single one
df = df[[len(t)<180 for t in df.text.values]]
df = df[[len(t)>50 for t in df.text.values]]

"""This is where you can change the name of the pickle file"""
with open('./tweets.pickle', 'rb') as f:
    char2int, int2char = pickle.load(f)

len_vocab = len(char2int)

model = Sequential()
model.add(Embedding(len_vocab, 64)) # , batch_size=batch_size
model.add(LSTM(64, return_sequences=True)) # , stateful=True
model.add(TimeDistributed(Dense(len_vocab, activation='softmax')))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')



"""This is where you can change the name of the saved model"""
model.load_weights('trump_model2.h5')
"""======================================================="""


print("\nGenerating 30 random tweets:\n")

for j in range(30):
    sentence = []
    letter = [char2int["<GO>"]] #choose a random letter
    for i in range(150):
        sentence.append(int2char[letter[-1]])
        if sentence[-1]=="<END>":
            break
        p = model.predict(np.array(letter)[None,:])
        letter.append(np.random.choice(len(char2int),1,p=p[0][-1])[0])
    print("".join(sentence))
    print("="*100)


while True:
    userInput = input("\nWhat do you want Donald Trump to say? ")
    letter = [char2int[letter] for letter in str(userInput)]
    sentence = [int2char[l] for l in letter]

    for i in range(150):
        if sentence[-1]=="<END>":
            break
        p = model.predict(np.array(letter)[None,:])
        letter.append(np.random.choice(len(char2int),1,p=p[0][-1])[0])
        sentence.append(int2char[letter[-1]])
    print("".join(sentence))