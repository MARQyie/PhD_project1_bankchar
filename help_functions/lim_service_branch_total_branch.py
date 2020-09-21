# Multiple functions to be used with SOD data (Summary of deposits)
''' This script contains multiple functions that calculate instrumental variables
    for working paper 1. 
    
    LimitedServiceBranches
    This function reads in the summary of deposits data and returns a pandas
    dataframe with IDRSSD, date, the total number of branches and the number 
    of limited and service branches per total branches
    
    statesActive
    This function calculates the number of states the bank is active. 
    
    noBranchBank
    This function extracts the variable UNIT from the SOD and returns it as a
    long panel df. Can be used to add as variable to the call report data
    
    meanMedStdDistance
    This method calculates the mean, median, standard deviation, maximum,
    the minimum of the distances between the bank branch and its head quarter. 

    Packages needed: Pandas, Numpy
    NOTE: This function replaces numberOfBranches
    '''
    
# Import Packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\Data\Data_SOD')

# Function to load the data
def readSODFiles(use_cols,begin_year,end_year):
    '''This method reads in all the data from the Summary of Deposits and returns
        a dataframe with the selected columns only. Requires numba,jit'''
    path = r'X:/My Documents/Data/Data_SOD/'
    df_sod = pd.DataFrame()
    
    #Read data  
    for i in range(begin_year,end_year):
        df_load = pd.read_csv(path + 'ALL_{}.csv'.format(i), encoding = 'ISO-8859-1', usecols = use_cols)
        df_sod = df_sod.append(df_load)
    
    return (df_sod)

#--------------------------------------------
# Function LimitedServiceBranches
def LimitedServiceBranches(begin_year, end_year):

    ## List of numbers for full and limited service branches
    full = range(11,13 + 1)
    limited = range(21,30 + 1)
    
    #Read data
    use_cols = ['YEAR','RSSDID','BRSERTYP'] 
    df_sod = readSODFiles(use_cols,begin_year,end_year)
    
    # Rename cols
    df_sod.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Count the number of branches
    num_branches = df_sod.groupby(['IDRSSD','date'])['IDRSSD'].agg('count')
    
    # Count number of full and limited service branches
    perc_full = df_sod[df_sod.BRSERTYP.isin(full)].groupby(['IDRSSD','date'])\
                ['IDRSSD'].agg('count').divide(num_branches)
    perc_limited = df_sod[df_sod.BRSERTYP.isin(limited)].groupby(['IDRSSD','date'])\
                ['IDRSSD'].agg('count').divide(num_branches).replace(np.nan,0.0)
    
    # Make a df of the three series
    df = pd.DataFrame(num_branches, index = num_branches.index)
    df.rename({'IDRSSD':'num_branch'}, axis = 1, inplace = True)
    df['perc_full_branch'], df['perc_limited_branch'] = perc_full, perc_limited
    
    return(df)

#-----------------------------------------------------
# Function statesActive
def statesActive(begin_year, end_year): 
    
    #Read data
    use_cols = ['YEAR','RSSDID','STALPBR'] 
    df_sod = readSODFiles(use_cols,begin_year,end_year)

    # Rename cols
    df_sod.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Count unique number of states
    unique_states = df_sod.groupby(['IDRSSD','date'])['STALPBR'].nunique()
    
    return(unique_states)
    
#-----------------------------------------------------
# Function noBranchBank
def noBranchBank(begin_year, end_year): 
    
    #Read data
    use_cols = ['YEAR','RSSDID','UNIT'] 
    df_sod = readSODFiles(use_cols,begin_year,end_year)

    # Rename cols
    df_sod.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Count unique number of states
    no_branch = df_sod.groupby(['IDRSSD','date'])['UNIT'].sum()
    
    return(no_branch)
    
#--------------------------------------------------
# Function meanMedStdDistance
def meanMedStdDistance(calculation = 'mean'):
    ''' 5 possible values for calculation: 1) mean, 2) median, 3) std, 4) max,
        5) min
    '''
    #Read the df
    path = r'X:/My Documents/Data/Data_SOD/'
    dataframe = pd.read_csv(path + 'SOD_all_distances.csv', index_col = 0)
    
    # Rename cols
    dataframe.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    if calculation == 'mean':
        return (dataframe.groupby(['IDRSSD','date'])['distance'].mean())
    elif calculation == 'median':
        return (dataframe.groupby(['IDRSSD','date'])['distance'].median())
    elif calculation == 'std':
        return (dataframe.groupby(['IDRSSD','date'])['distance'].std())
    elif calculation == 'max':
        return (dataframe.groupby(['IDRSSD','date'])['distance'].max())
    elif calculation == 'min':
        return (dataframe.groupby(['IDRSSD','date'])['distance'].replace(0.0, np.nan).min().fillna(0.0))
    
