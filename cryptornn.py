import numpy as np
import pandas as pd
from pip import main

#Some parameters
SEQUENCE_LEN = 60  # Sequence of 60 minutes
CRY_2_PRED = "ETH-USD"
FUTURE_PRED = 3    # Predict the next 3 minutes

def Decide(current, future): # Decides whther the price increases or decreases
    return int(float(current) < float(future))

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

#print(main_df.size)
#print(main_df[[f"{CRY_2_PRED}-close", "future", "target"]].head(10))

#Separate the data to train and validation
times = sorted(main_df.index.values)
last5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last5pct)]
main_df = main_df[(main_df.index < last5pct)]






