import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import LOCATION_CHOOSEN, OUTPUT_DIR, DATA_CACHE_DIR, STRINGENCY_BASED_GDP, OPTIMAL_VALUES_FILE, MODELS_DIR, RL_LEARNING_TYPE

data_path = os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + ".csv")
if os.path.exists(data_path):
    df_covid = pd.read_csv(data_path)
    df_covid = df_covid[["date", "stringency_index", "total_cases", "total_deaths", "total_vaccinations", "population", "people_fully_vaccinated"]]
    
    # TODO: figure out if you want to do this
    # df_covid.interpolate(method='time', inplace=True)
    df_covid.ffill(inplace=True)
    df_covid.bfill(inplace=True)
    df_covid['date'] = pd.to_datetime(df_covid['date'])

data_path = os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + "_with_GDP.csv")
if os.path.exists(data_path):
    df_gdp = pd.read_csv(data_path)
    df_gdp = df_gdp[["date", " Gross Domestic Product (GDP)  Normalised"]]
    df_gdp['date'] = pd.to_datetime(df_gdp['date'])


print(df_covid, df_gdp)

# Set 'time' as index for both DataFrames
df_covid.set_index('date', inplace=True)
df_gdp.set_index('date', inplace=True)

# Merge the two DataFrames
df = pd.concat([df_covid, df_gdp], axis=1)

# Interpolate the missing data
df.interpolate(method='time', inplace=True)

print(df)

df.to_csv(os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + "_merged_data.csv"))