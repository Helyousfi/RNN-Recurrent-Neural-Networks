from random import random
import numpy as np
import pandas as pd
from pip import main
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import deque
import time, random

#Some parameters
SEQUENCE_LEN = 60  # Sequence of 60 minutes
CRY_2_PRED = "ETH-USD"
FUTURE_PRED = 3    # Predict the next 3 minutes

def Decide(current, future): # Decides whther the price increases or decreases
    return int(float(current) < float(future))

def Preprocess_dataframe(df):
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace  =True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    days_prev = deque(maxlen=SEQUENCE_LEN)

    for value in df.values:
        days_prev.append([n for n in value[:-1]])
        if len(days_prev) == SEQUENCE_LEN:
            sequential_data.append([np.array(days_prev), value[-1]])


    return random.shuffle(sequential_data)


#Loading the dataset and  creating the main dataframe
main_df = pd.DataFrame()
cryptos = ["BCH-USD", "BTC-USD", "ETH-USD", "LTC-USD"]
for crypto in cryptos:
    dataset = f"datasets/crypto_data/{crypto}.csv"
    df = pd.read_csv(dataset, names = ["time", "low", "high", "open", "close", "volume"])
    df.set_index(keys="time", inplace=True)
    
    main_df[f"{crypto}-close"] = df["close"]
    main_df[f"{crypto}-volume"] = df["volume"]

main_df.fillna(method="ffill", inplace = True)
main_df.dropna(inplace = True)

#Shift the ratio to predict
main_df["future"] = main_df[f"{CRY_2_PRED}-close"].shift(-FUTURE_PRED)
main_df["target"] = list(map(Decide, main_df[f"{CRY_2_PRED}-close"], 
                            main_df["future"]))
main_df.drop("future", 1, inplace=True)


print(main_df.head(30))
#print(main_df[[f"{CRY_2_PRED}-close", "future", "target"]].head(10))

#Separate the data to train and validation
times = sorted(main_df.index.values)
last5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last5pct)]
main_df = main_df[(main_df.index < last5pct)]
Preprocess_dataframe(main_df)

print(main_df["BCH-USD-close"].iloc[0])



