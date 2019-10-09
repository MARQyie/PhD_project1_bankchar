#------------------------------------------
# Add Macro and structural variables to dataset working paper
# Mark van der Plaat
# October 2019 

   
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

from datetime import timedelta

#-------------------------------------------
# Set file path and names
path = r'X:/My Documents/Data/Data_FRED/'

file_gdp = r'GDP.csv'
file_fedfunds = r'FEDFUNDS.csv'

#-------------------------------------------
# Load the data
df_gdp = pd.read_csv(path + file_gdp)
df_fed = pd.read_csv(path + file_fedfunds)

## Set datetime
df_gdp.DATE = pd.to_datetime(df_gdp.DATE) - timedelta(days = 1)
df_fed.DATE = pd.to_datetime(df_fed.DATE) - timedelta(days = 1)

## Set index
df_gdp.set_index('DATE', inplace = True)
df_fed.set_index('DATE', inplace = True)

#-------------------------------------------
# Make data yearly
## GDP - Take all last quarters 
df_gdp = df_gdp[df_gdp.index.month == 12]
df_gdp.index = df_gdp.index.year

## Fed funds - Take average
df_fed = df_fed.groupby(pd.PeriodIndex(df_fed.index, freq = 'Y'), axis = 0).mean()
df_fed.index = df_fed.index.to_timestamp().year

#-------------------------------------------
# Make new df with both series
df_mac = pd.concat([df_gdp, df_fed], axis=1).dropna()

#-------------------------------------------
# Add variables
## GDP growth
df_mac['gdp_growth'] = df_mac.GDP.pct_change().multiply(1e2) 

## Dummy Dodd-Frank Act
df_mac['doddfrank'] = (df_mac.index >= 2010) * 1

#--------------------------------------------
# Add to the total df
## Load the df
df_bank = pd.read_csv('df_wp1_clean.csv', index_col = 0)

### Make multi index
df_bank.date = pd.to_datetime(df_bank.date.astype(str).str.strip() + '1230').dt.year

## Merge the dfs
df = df_bank.merge(df_mac, how = 'left', left_on = df_bank.date, right_on = df_mac.index)

#--------------------------------------------
# Save the df

# Drop key_0 column
df.drop(columns = ['key_0'], inplace = True)

df.to_csv('df_wp1_clean_macro.csv')