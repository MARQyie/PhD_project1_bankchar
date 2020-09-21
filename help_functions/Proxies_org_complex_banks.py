# Multiple functions to be used with SOD data (Summary of deposits)
''' This script contains multiple functions that calculate instrumental variables
    for working paper 1. 
    
    LimitedServiceBranches
    This function reads in the summary of deposits data and returns a pandas
    dataframe with IDRSSD, date, the total number of branches and the number 
    of limited and service branches per total branches
    
    spatialComplexity
    This function calculates the number of states and OCC districts the bank
    is active. 
    
    noBranchBank
    This function extracts the variable UNIT from the SOD and returns it as a
    long panel df. Can be used to add as variable to the call report data
    
    maxDistanceBranches
    This method calculates the  maximum, distances between any branch. 

    Packages needed: Pandas, Numpy
    NOTE: This function replaces numberOfBranches
    '''
    
# Import Packages
import pandas as pd
import numpy as np

import os
#os.chdir(r'X:\My Documents\Data\Data_SOD')
os.chdir(r'D:\RUG\Data\Data_SOD')

# Function to load the data
def readSODFiles(use_cols,begin_year,end_year):
    '''This method reads in all the data from the Summary of Deposits and returns
        a dataframe with the selected columns only. Requires numba,jit'''
#    path = r'X:/My Documents/Data/Data_SOD/'
    path = r'D:/RUG/Data/Data_SOD/'
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
    
#-----------------------------------------------------
# Function spatialComplexity
def spatialComplexity(begin_year, end_year): 
    ''' Functions that returns the number of unique states active and OCC district
        per bank-year. 
        
        Output: Pandas DataFrame
        '''
    
    #Read data
    use_cols = ['YEAR','RSSDID','STALPBR','OCCDIST'] 
    df_sod = readSODFiles(use_cols,begin_year,end_year)

    # Rename cols
    df_sod.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Count unique states and OCC districts
    unique_occ = df_sod.groupby(['IDRSSD','date'])['OCCDIST'].nunique()
    unique_states = df_sod.groupby(['IDRSSD','date'])['STALPBR'].nunique()

    # Make a df of the two series
    df = pd.DataFrame(unique_occ, index = unique_occ.index)
    df.rename({'IDRSSD':'unique_occ'}, axis = 1, inplace = True)
    df['unique_states']= unique_states
    
    return(df)
    
#--------------------------------------------------
# Function maxDistanceBranches
def maxDistanceBranches():
    ''' Calculates the maximum distance between the branches of a bank.
        
        Difference meanMedStdDistance: need not necessarily run via the 
        headquarters    
        '''
    from sklearn.neighbors import DistanceMetric
    from numpy import radians
    
    #Read the df
    path = r'X:/My Documents/Data/Data_SOD/'
    dataframe = pd.read_csv(path + 'SOD_all_distances.csv', index_col = 0)
    
    # Rename cols
    dataframe.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Define the max distance calculator
    def distanceCalculator(list_coordinates):
        ''' Calculates the maximum distance between the coordinates in the list
            with the havesine formula. 
            
            Input : list'''
            
        # Initialize the operator    
        dist = DistanceMetric.get_metric('haversine')
        
        # Calculate the distances
        R = 6372.8  # Radius of the earth
        dist_array = dist.pairwise(radians(list_coordinates)) * R  
        
        if dist_array.size == 1.0:
            return (0.0)
        else:
            return (np.max(dist_array))
    
    return(dataframe.groupby(['IDRSSD','date'])[['branch_lat', 'branch_lng']].apply(distanceCalculator)) 

''' OLD FUNCTIONS
def HHIState(begin_year, end_year): 
   
    #Read data
    use_cols = ['YEAR','RSSDID','STALPBR'] 
    df_sod = readSODFiles(use_cols,begin_year,end_year)

    # Rename cols
    df_sod.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Function to calculate the inverse HHI
    def invHHI(series):
        #prelims
        n = series.shape[0]
        
        # Partial calculations
        if n == 1.0:
            return(0.0)
        else:
            hhi = np.sum(series.value_counts(normalize = True, dropna = False) ** 2)
            frac = 1 / (1 - (1 / n))
        
        return(frac * (1 - hhi))
    
    hhi_states = df_sod.groupby(['IDRSSD','date'])['STALPBR'].apply(invHHI)

    return(hhi_states)

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
# Function OCCActive
def OCCActive(begin_year, end_year): 
    
    #Read data
    use_cols = ['YEAR','RSSDID','STALPBR'] 
    df_sod = readSODFiles(use_cols,begin_year,end_year)

    # Rename cols
    df_sod.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Count unique OCC districts
    unique_occ = df_sod.groupby(['IDRSSD','date'])['OCCDIST'].nunique()
    
    return(unique_states)
       
#--------------------------------------------------
# Function meanMedStdDistance
def meanMedStdDistance(calculation = 'mean'):
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
        '''

