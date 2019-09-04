#------------------------------------
# This code is used to check some characteristics of US banks

#import packages
import pandas as pd
import numpy as np 
import os
os.chdir('X:\My Documents\PhD\Courses\Applied_macroeconometrics\Assignments\Tom_Boot')
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale= 1 )

#------------------------------------
## Set parameters
start = 2001
end = 2019
path = r'X:/My Documents/Data/Data_call_reports_FFIEC2'

#------------------------------------
### Setup data
## Securitization data
df = pd.DataFrame()

for i in range(start,end):
    '''Parses over all available datasets, selects the securitization variables and appends them to the dataframe. '''
    df_load = pd.read_csv(path + '/{}/FFIEC CDR Call Schedule RCS 1231{}.txt'.format(i,i), \
                 sep='\t',  skiprows = [1])
    df_load = df_load[['IDRSSD','RCFDB705','RCFDB711','RCONB705','RCONB711']]
    df_load['date'] = int('{}'.format(i))
    df = df.append(df_load)

# Combine variables
df['RCB705'] = df.apply(lambda x: x.RCFDB705 if x.RCFDB705 >= 0.0 else (x.RCONB705), axis = 1) 
df['RCB711'] = df.apply(lambda x: x.RCFDB711 if x.RCFDB711 >= 0.0 else (x.RCONB711), axis = 1) 
df['RCBtot'] = df.apply(lambda x: x.RCB705 + x.RCB711, axis = 1) 

## Bank state information  
df_bank = pd.DataFrame()    
    
for i in range(start,end):
    df_load = pd.read_csv(path + '/{}/FFIEC CDR Call Bulk POR 1231{}.txt'.format(i,i), \
                     sep='\t')    
    df_load = df_load[['IDRSSD','Financial Institution Name', 'Financial Institution State']]
    df_load['date'] = int('{}'.format(i))
    df_bank = df_bank.append(df_load)   

# Renames state column 
df_bank.rename(columns = {'Financial Institution Name':'name', 'Financial Institution State':'state'} ,inplace = True)

# Join the two datasets
# Note: includes non-deposit insurance as well
df = df.merge(df_bank, on = ['date', 'IDRSSD'], how = 'left')

## Add balance sheet securities 
df_bal = pd.DataFrame()    
    
for i in range(start,end):
    df_load = pd.read_csv(path + '/{}/FFIEC CDR Call Schedule RC 1231{}.txt'.format(i,i), \
                     sep='\t',skiprows = [1])    
    df_load = df_load[['IDRSSD','RCFD3545','RCFD3548','RCON3545','RCON3548']]
    df_load['date'] = int('{}'.format(i))
    df_bal = df_bal.append(df_load) 
    
df_bal['RC3545'] = df_bal.apply(lambda x: x.RCFD3545 if x.RCFD3545 >= 0.0 else (x.RCON3545), axis = 1) 
df_bal['RC3548'] = df_bal.apply(lambda x: x.RCFD3548 if x.RCFD3548 >= 0.0 else (x.RCON3548), axis = 1) 
df_bal['RC35tot'] = df_bal.apply(lambda x: abs(x.RC3545) + abs(x.RC3548), axis = 1)   

# Join the two datasets
# Note: includes non-deposit insurance as well
df = df.merge(df_bal, on = ['date', 'IDRSSD'], how = 'left')

## Fill NaNs
df.fillna(value = 0, inplace = True)

## Check Merge
df.info() #Went OK

## Drop non-mainland states and zero
non_states = ('AS','FM','GU','MH','MP','PW','PR','VI','0','0 ')
df = df[~df.state.isin(non_states)]

#----------------------------------
### Analysis states
## In which states are the non-securitizing banks?
df_ns = df[df.RCBtot == 0]
state_count_ns = df_ns.state.value_counts() # In all 51 states (incl DC)
state_names_ns = df_ns.state.unique()

## In which states are the securitizing banks?
df_s = df[df.RCBtot != 0]
state_count_s = df_s.state.value_counts() # in 47 states
state_names_s = df_s.state.unique()

## In which 4 states are no securitizers whatsoever?
states_intersect = state_names_ns[~np.isin(state_names_ns, state_names_s)]
print(states_intersect) # Not surprising?

# How many obs?
num_ns_states = df_ns[df_ns.state.isin(states_intersect)] # 1192

#----------------------------------
### Simple test to check whether non-securitizing banks have access to capital markets
## Make dummy: 1 if RC35tot > 0
df_ns['Dcm'] = (df_ns.RC35tot > 0) * 1 
print('Number of obs with access to capital markets: {}'.format(df_ns.Dcm.sum()))

num_ns_states2 = df_ns[df_ns.state.isin(states_intersect)]
