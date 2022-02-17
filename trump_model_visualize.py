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



print("\n====================Visualization====================\n")

tweets_over_time = df.groupby([df.index.year]).count()
fig = plt.figure()
fig.suptitle("Tweets Over Time")
tweets_over_time["id"].plot(kind="bar", figsize=(11, 5))
plt.xlabel("Time")
plt.ylabel("Tweets (per year)")
plt.show()

df.plot(kind="line", figsize=(11, 5), y="retweets")
df.plot(kind="line", figsize=(11, 5), y="favorites")
plt.show()

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

text = df.text.values
wordcloud = WordCloud(width = 3000, height = 2000, background_color = "black", stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(figsize = (12, 8), facecolor = "k", edgecolor = "k")
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()