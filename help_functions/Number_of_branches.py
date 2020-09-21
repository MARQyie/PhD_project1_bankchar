# Number of Branches (Summary of deposits)
''' This function reads in the summary of deposits data and returns a pandas
    dataframe with IDRSSD, date and number of branches

    Packages needed: Pandas
    '''
    
# Import Packages
import pandas as pd

# Function
def numberOfBranches(begin_year, end_year):
    # Setup
    path = r'X:/My Documents/Data/Data_SOD/'
    df_sod = pd.DataFrame()
    
    #Read data
    for i in range(begin_year,end_year):
        df_load = pd.read_csv(path + 'ALL_{}.csv'.format(i), encoding = 'ISO-8859-1')
        df_sod = df_sod.append(df_load)
    
    # Rename cols
    df_sod.rename({'YEAR':'date', 'RSSDID':'IDRSSD'}, axis = 1, inplace = True)
    
    # Count the number of branches
    num_branches = df_sod.groupby(['date','IDRSSD'])['IDRSSD'].agg('count')
    
    # Rename col
    num_branches = pd.DataFrame(num_branches, index = num_branches.index)
    num_branches.rename({'IDRSSD':'num_branch'}, axis = 1, inplace = True)
    
    return(num_branches)
    
    