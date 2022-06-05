from random import random
import numpy as np
import pandas as pd
from pip import main
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import deque
import time, random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import keras

#Some parameters
SEQUENCE_LEN = 60  # Sequence of 60 minutes
CRY_2_PRED = "ETH-USD"
FUTURE_PRED = 3    # Predict the next 3 minutes
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQUENCE_LEN}-SEQ-{FUTURE_PRED}-PRED-{int(time.time())}"

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
    random.shuffle(sequential_data)
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y


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

train_x, train_y = Preprocess_dataframe(main_df)
validation_x, validation_y = Preprocess_dataframe(validation_main_df)

print(f"Train data : {len(train_x)} Validation data : {len(validation_x)}")
print(f"Train data buys : {train_y.count(0)} Train data dont buys : {train_y.count(0)}")
print(f"Validation data buys : {validation_y.count(0)} Validation data dont buys : {validation_y.count(0)}")



model = Sequential()
model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(128, input_shape = (train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

"""
# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)
"""

model = keras.models.load_model("saved_model.pb")