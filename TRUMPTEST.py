import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, TimeDistributed
from keras.models import load_model, model_from_json
import pickle

df = pd.read_csv("realdonaldtrumpedited.csv",parse_dates=True) # might need to change location if on Floydhub
print(df.text)
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
print(df.text)