# Test: difference quarterly call reports and year-end
# Only check the securitization variables

#------------------------------------------------------------
# Import Packages
#------------------------------------------------------------
    
import pandas as pd
import numpy as np 

import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization
num_cores = mp.cpu_count()

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

import csv

#------------------------------------------------------------
# Set prelims
#------------------------------------------------------------

# Parse over the following dates:
dates = ['03312011','06302011','06302011','12312011']

#------------------------------------------------------------
# Set file paths/names and variable names
#------------------------------------------------------------

#------------------------------------------
# Set file paths
path_call = r'D:/RUG/Data/Data_call_reports_FFIEC2'

#------------------------------------------
# Set filenames
file_rcs = r'/2011/FFIEC CDR Call Schedule RCS {}.txt'

vars_rcs = '|'.join(['IDRSSD','B705','B706','B707','B708','B709','B710','B711',\
                     'B790','B791','B792','B793','B794','B795','B796',\
                     'B712','B713','B714','B715','B716','B717','B718',\
                     'C393','C394','C395','C396','C397','C398','C399',\
                     'C400','C401','C402','C403','C404','C405','C406'])  
    
#------------------------------------------------------------
# Set functions
#------------------------------------------------------------
def loadGeneral(i, file, var_list):
    ''' A General function for loading call reports data, no breaks '''
    global path_call
    
    df_load = pd.read_csv((path_call + file).format(i), sep='\t',  skiprows = [1,2])
    df_load['date'] = i  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])

def combineVars(data, elem):
    ''' Function to combine RCFD and RCON into one variable '''
    data['RC' + elem] = data['RCFD' + elem].fillna(data['RCON' + elem])
    
    return(data['RC' + elem])

#------------------------------------------
## Load rcs data 
if __name__ == '__main__':
    df_rcs = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcs, vars_rcs) for i in dates))
    
var_num = ['B705','B706','B707','B708','B709','B710','B711',\
           'B790','B791','B792','B793','B794','B795','B796',\
           'B712','B713','B714','B715','B716','B717','B718',\
           'C393','C394','C395','C396','C397','C398','C399',\
           'C400','C401','C402','C403','C404','C405','C406']

if __name__ == '__main__':
    df_rcs_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rcs, elem) for elem in var_num), axis = 1)

cols_remove =  [col for col in df_rcs.columns if not col[4:] in var_num]      
df_rcs = pd.concat([df_rcs[cols_remove], df_rcs_combvars], axis = 1)

#------------------------------------------------------------
# Check Securitization
#------------------------------------------------------------

# Set Series
sec = pd.DataFrame(df_rcs.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
               'RCB711']].sum(axis = 1, skipna = True), columns = ['sec'])

## add date and IDRSSD
for var in ('IDRSSD','date'):
    sec[var] = df_rcs[var]

# Dropna    
sec.dropna(inplace = True)
    
# Take the average per unique bank
sec_mean = sec.groupby(['IDRSSD']).mean()

# Select year ends
sec_ye = sec[sec.date == '12312011']

# Check N unique securitizers per dataset
nunique_sec_mean = sec_mean[sec_mean.sec > 0].index.get_level_values(0).nunique()
nunique_sec_ye = sec_ye[sec_ye.sec > 0].IDRSSD.nunique()

#------------------------------------------------------------
# Check Max Credit Exposure
#------------------------------------------------------------

ce = pd.DataFrame(df_rcs.loc[:,['RCB712','RCB713','RCB714','RCB715','RCB716','RCB717','RCB718',\
           'RCC393','RCC394','RCC395','RCC396','RCC397','RCC398','RCC399',\
           'RCC400','RCC401','RCC402','RCC403','RCC404','RCC405','RCC406']].sum(axis = 1, skipna = True), columns = ['ce'])

## add date and IDRSSD
for var in ('IDRSSD','date'):
    ce[var] = df_rcs[var]

# Dropna    
ce.dropna(inplace = True)
    
# Take the average per unique bank
ce_mean = ce.groupby(['IDRSSD']).mean()

# Select year ends
ce_ye = ce[ce.date == '12312011']

# Check N unique securitizers per dataset
nunique_ce_mean = ce_mean[ce_mean.ce > 0].index.get_level_values(0).nunique()
nunique_ce_ye = ce_ye[ce_ye.ce > 0].IDRSSD.nunique()

''' CONCLUSION: Averaging the quarterly data like Casu et al. (2013) adds
    about 10% of observations to the data. This might be usefull. It does
    not lead to a huge increase in observations for securitization or maximum
    credit exposure.
    ''' 