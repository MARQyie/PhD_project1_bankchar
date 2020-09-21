#------------------------------------------
# Setup the DataFrame for wp1
# Mark van der Plaat
# Sep 2019; data update: Mar 2020 

''' 
    This document partially cleans the the dataset and add variables 
    for further analyses for working paper #1

    Data used: US Call reports 2001-2018 year-end
    Only insured, commercial banks are taken
    ''' 
#------------------------------------------------------------
# Import Packages
#------------------------------------------------------------
    
import pandas as pd
import numpy as np

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

#import sys # to use the help functions needed
#sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

#------------------------------------------------------------
# Load Data
#------------------------------------------------------------
df_raw = pd.read_csv('Data\df_wp1_raw.csv')

#------------------------------------------------------------
# Filter Data
#------------------------------------------------------------
'''Only select insured, commercial banks with a physical location in the 50 states
    of the US.'''

## Select insured and commercial banks
'''RSSD9048 == 200, RSSD9424 != 0'''
df_filtered = df_raw[(df_raw.RSSD9048 == 200) & (df_raw.RSSD9424 != 0)]

## Only take US states, no territories
''' Based on the state codes provided by the FRB'''
df_filtered = df_filtered[df_filtered.RSSD9210.isin(range(1,57))]

#------------------------------------------
# Drop all banks with three years or fewer of observations
drop_banks = df_filtered.IDRSSD.value_counts()[df_filtered.IDRSSD.value_counts() <= 3].index.unique().tolist()
df_filtered = df_filtered[~df_filtered.IDRSSD.isin(drop_banks)]

#------------------------------------------
# Drop missings in Total assets, loans, deposits, income
nandrop_cols = ['RC2170','RC2122','RC2200','RIAD4340']
df_filtered.dropna(subset = nandrop_cols , inplace = True)

#------------------------------------------
# Drop negative and zero values in Total assets and loans
df_filtered = df_filtered[(df_filtered.RC2170 > 0) & (df_filtered.RC2122 > 0)]

#------------------------------------------------------------
# Create new df and variabless
#------------------------------------------------------------

#------------------------------------------
# Create new df

df = df_filtered[['IDRSSD','date']]

#------------------------------------------
# Make new variables

# Dependent Variables
## Total Charge-offs
df['net_coff_tot'] = (df_filtered[['RIADB747','RIADB748','RIADB749','RIADB750',\
  'RIADB751','RIADB752','RIADB753', 'RIAD4635']].sum(axis = 1) - \
df_filtered[['RIADB754','RIADB755','RIADB756','RIADB757','RIADB758','RIADB759','RIADB760', 'RIAD4605']].\
sum(axis = 1)) / df_filtered.RC2170

## Non-performing loans
#df['npl']
#TODO

# Control variables Baseline
## Loan sales
df.loc[:,'ls_tot'] = df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
               'RCB711','RCB790','RCB791','RCB792','RCB793','RCB794','RCB795',\
                  'RCB796']].sum(axis = 1, skipna = True) / df_filtered.RC2170
          
## Regulatory Leverage Ratio
df.loc[:,'reg_lev'] = df_filtered.RC7204

## Loan Ratio
df['loanratio'] = df_filtered.RC2122 / df_filtered.RC2170

## ROA
df['roa'] = df_filtered.RIAD4340 / df_filtered.RC2170

## Deposit Ratio
df['depratio'] = df_filtered.RC2200 / df_filtered.RC2170

## Commercial Loan Ratio
df['comloanratio'] = df_filtered['RCON1766'] / df_filtered.RC2122

## Mortgage Ratio
df_filtered['loans_sec_land'] = df_filtered.apply(lambda x: x.RCON1415 if ~np.isnan(x.RCON1415) else x[['RCF158','RCF159']].sum(), axis = 1) # Not needed in analysis
df_filtered['loans_sec_nonfarm'] = df_filtered.apply(lambda x: x.RCON1480 if ~np.isnan(x.RCON1480) else x[['RCF160','RCF161']].sum(), axis = 1) # Not needed in analysis
df['mortratio'] = df_filtered[['loans_sec_land','RC1420','RC1460','RC1797',\
  'RC5367','RC5368','loans_sec_nonfarm']].sum(axis = 1).divide(df_filtered.RC2122)

## Consumer Loan Ratio
df['consloanratio'] = df_filtered[['RCB538','RCB539','RC2011','RCK137','RCK207']].sum(axis = 1).divide(df_filtered.RC2122)

## Loans HHI
df['agriloanratio'] = df_filtered.RC1590 / df_filtered.RC2122 # Not needed in analysis, but needed to calculate loan HHI
df['loanhhi'] = df[['mortratio','consloanratio','comloanratio','agriloanratio']].pow(2).sum(axis = 1) 

## Cost-to-income (costs / operating income = costs / (net int. inc + non-int inc + gains on sec. - non-int. exp.))
df['costinc'] = (df_filtered.RIAD4093 / (df_filtered.RIAD4074 + df_filtered.RIAD4079 + df_filtered.RIAD3521 + df_filtered.RIAD3196 - df_filtered.RIAD4093)).replace(np.inf, np.nan)

## Size
df['size'] = np.log(df_filtered.RC2170)

## BHC Indicator
df['bhc'] = (round(df_filtered.RSSD9364) > 0.0) * 1

# Variables for instruments
from Code_docs.help_functions.Proxies_org_complex_banks import LimitedServiceBranches,\
     spatialComplexity, noBranchBank, readSODFiles

## Number of limited service, full service and total branches
df_branches = LimitedServiceBranches(2001,2020)
df_branches.reset_index(inplace = True)
df_branches['log_num_branch'] = np.log(df_branches.num_branch)

## Number of States Active
df_complex = spatialComplexity(2001,2020)
df_complex = df_complex.reset_index()
df_complex['log_states'] = np.log(df_complex.unique_states)

## No Bank Branches
df_nobranch = noBranchBank(2001,2020)
df_nobranch = df_nobranch.reset_index()

## Add the dfs to the dataframe
df = df.merge(df_branches, on = ['IDRSSD', 'date'], how = 'left')
df = df.merge(df_complex, on = ['IDRSSD', 'date'], how = 'left')
df = df.merge(df_nobranch, on = ['IDRSSD', 'date'], how = 'left')

## Drop banks that have no known number of branches
df.dropna(subset = ['num_branch'], inplace = True) 

## Employees and log Employees
df['empl'] = df_filtered.RIAD4150
df['log_empl'] = np.log(df_filtered.RIAD4150 + 1)

# Variables for Robustness
## Securitization
df['ls_sec'] = df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
               'RCB711']].sum(axis = 1, skipna = True)\
                / df_filtered.RC2170
df['ls_nonsec'] = df_filtered.loc[:,['RCB790','RCB791','RCB792','RCB793','RCB794','RCB795',\
                  'RCB796']].sum(axis = 1, skipna = True) / df_filtered.RC2170
  
## Dummy Loan sales
df['ls_dum'] = (df.ls_tot > 0.0) * 1
  
## Credit Exposure
df['credex_sec'] = df_filtered[['RCB712','RCB713','RCB714','RCB715','RCB716','RCB717','RCB718',\
                                  'RCB719','RCB720','RCB721','RCB722','RCB723','RCB724','RCB725',\
                                  'RCC393','RCC394','RCC395','RCC396','RCC397','RCC398','RCC399',\
                                  'RCC400','RCC401','RCC402','RCC403','RCC404','RCC405','RCC406',\
                                  'RCHU09','RCFDHU10','RCFDHU11','RCFDHU12','RCFDHU13','RCFDHU14','RCHU15']].\
                          sum(axis = 1, skipna = True) / df_filtered.RC2170
df['credex_nonsec'] = df_filtered[['RCB797','RCB798','RCB799','RCB800','RCB801','RCB802','RCB803']].\
                          sum(axis = 1, skipna = True) / df_filtered.RC2170
df['credex_tot'] = df_filtered[['RCB712','RCB713','RCB714','RCB715','RCB716','RCB717','RCB718',\
                                  'RCB719','RCB720','RCB721','RCB722','RCB723','RCB724','RCB725',\
                                  'RCC393','RCC394','RCC395','RCC396','RCC397','RCC398','RCC399',\
                                  'RCC400','RCC401','RCC402','RCC403','RCC404','RCC405','RCC406',\
                                  'RCHU09','RCFDHU10','RCFDHU11','RCFDHU12','RCFDHU13','RCFDHU14',\
                                  'RCHU15''RCB797','RCB798','RCB799','RCB800','RCB801','RCB802','RCB803']].\
                          sum(axis = 1, skipna = True) / df_filtered.RC2170

'''NOTE: Any variables for figures could be taken from the filtered raw data 
    and the final data'''

#------------------------------------------------------------
# Drop any nans in baseline variables
#------------------------------------------------------------
vars_baseline = ['net_coff_tot','npl','ls_tot','reg_lev','loanratio','roa','depratio',\
                 'comloanratio','mortratio','loanhhi','costinc','size','bhc','log_empl']
df.dropna(subset = vars_baseline, inplace = True)             

#------------------------------------------
## Save df
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')
df_filtered.to_csv('Data\df_wp1_filtered.csv')
df.to_csv('Data\df_wp1_main.csv')

