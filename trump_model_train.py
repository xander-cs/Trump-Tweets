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


trump_tweets = [text for text in df.text.values[::-1]]
all_tweets = "".join(trump_tweets)
char2int = dict(zip(set(all_tweets), range(len(set(all_tweets)))))
char2int["<END>"] = len(char2int)
char2int["<GO>"] = len(char2int)
char2int["<PAD>"] = len(char2int)
int2char = dict(zip(char2int.values(), char2int.keys()))

print("Setting up LSTM...\n")

len_vocab = len(char2int)

model = Sequential()
model.add(Embedding(len_vocab, 64)) # , batch_size=batch_size
model.add(LSTM(64, return_sequences=True)) # , stateful=True
model.add(TimeDistributed(Dense(len_vocab, activation="softmax")))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

text_num = [[char2int["<GO>"]]+[char2int[c] for c in tweet]+ [char2int["<END>"]] for tweet in trump_tweets]

sentence_len = 40
num_examples = 0

for tweet in text_num:
    num_examples += len(tweet)-sentence_len

x = np.zeros((num_examples, sentence_len))
y = np.zeros((num_examples, sentence_len))
k = 0

for tweet in text_num:
    for i in range(len(tweet)-sentence_len):
        x[k,:] = np.array(tweet[i:i+sentence_len])
        y[k,:] = np.array(tweet[i+1:i+sentence_len+1])
        k += 1      

y = y.reshape(y.shape+(1,))


print("\n====================Training====================\n")

n_epochs = int(input("Amount of epochs to train: "))
for i in range(n_epochs+1):
    sentence = []
    letter = [char2int["<GO>"]] #choose a random letter
    for i in range(150):
        sentence.append(int2char[letter[-1]])
        p = model.predict(np.array(letter)[None,:])
        letter.append(np.random.choice(len(char2int),1,p=p[0][-1])[0])
    print("".join(sentence))
    print("="*100)
    if i!=n_epochs:
        model.fit(x,y, batch_size=1024, epochs=1)

saveModel = input("\nSave the model?")
if saveModel[0] == "y" or saveModel[0] == "Y":



    """This is where you can change the name of the saved model"""
    model.save('trump_model.h5')
    """======================================================="""


"""This is where you can change the name of the pickle file"""
with open('./tweets.pickle', 'wb') as f:
    pickle.dump((char2int, int2char), f)