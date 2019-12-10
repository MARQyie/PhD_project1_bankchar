#------------------------------------------
# Check the dfs on outliers etc.
# Mark van der Plaat
# December 2019 

 # Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

## Plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

#---------------------------------------------- 
#----------------------------------------------
# Prelims
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# Load data and add needed variables

# Load df
df = pd.read_csv('Data\df_wp1_newvars.csv', index_col = 0)

## Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1231')

## Vars that are possibly needed for analysis (No dummies)
vars_needed = ['provratio','rwata','net_coffratio_tot_ta',\
               'allowratio_tot_ta','ls_tot_ta','size',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170',\
               'num_branch', 'RIAD4150', 'perc_limited_branch',\
               'unique_states','distance','roe', 'nim']

#----------------------------------------------
# Drop Nans and check inf
## NA drop
df.dropna(subset = vars_needed, inplace = True)

## Inf check
### NOTE: One inf in provratio, rest is good
inf_list = np.sum(np.isinf(df[vars_needed]) * 1, axis = 0)
df.replace([np.inf, -np.inf], np.nan).dropna(subset = vars_needed, inplace = True)

#----------------------------------------------
# Dictionary with Variable Names
dict_var_names = {'distance':'Max Distance Branches',
                 'provratio':'Loan Loss Provisions',
                 'rwata':'RWA/TA',
                 'net_coffratio_tot_ta':'Loan Charge-offs',
                 'allowratio_tot_ta':'Loan Loss Allowances',
                 'ls_tot_ta':'Loan Sales/TA',
                 'dum_ls':'Dummy Loan Sales',
                 'size':'Log(TA)',
                 'RC7205':'Regulatory Capital Ratio',
                 'loanratio':'Loan Ratio',
                 'roa':'ROA',
                 'depratio':'Deposit Ratio',
                 'comloanratio':'Commercial Loan Ratio',
                 'RC2170':'Total Assets',
                 'num_branch':'Num Branches',
                 'bhc':'BHC Indicator',
                 'RIAD4150':'Num Employees',
                 'perc_limited_branch':'Limited Branches (in %)',
                 'perc_full_branch':'Full Branches (in %)',
                 'unique_states':'Num States Active',
                 'UNIT':'Unit Bank Indicator',
                 'roe':'ROE',
                 'nim':'Net Interest Margin'}


#----------------------------------------------
# Box plots
#----------------------------------------------

for var in vars_needed:
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(dict_var_names[var]))
    
    data = df[var]
    ax.boxplot(data)
    
    plt.xticks([1], ['Full Sample (Not Clean)'])
    
    fig.savefig('Figures\Plots_datacleaning\Scatter_{}_not_cleaned.png'.format(var))
    plt.clf()
    
''' NOTES: 
        One big outlier in num_branch and total assets (same bank/year)
        Limit ROA to [-1,1]
        Limit RC7205, loanratio to [0,1]
        Drop outlier on ls_tot_ta
        limit provratio to [-1,1]
        ROE is a mess, don't use
    '''
#-----------------------------------------------
# Clean data further
#----------------------------------------------

## Remove big outlier num_branch and TA
df = df[df.num_branch != df.num_branch.max()]

## Limit ROA and prov ratio to [-1,1]
vars_limit = ['roa','provratio']

for i in vars_limit:
    df = df[df['{}'.format(i)].between(-1,1,inclusive = True)]
    
# Limit RC7205 to [0,1] 
vars_limit = ['loanratio','RC7205']

for i in vars_limit:
    df = df[df['{}'.format(i)].between(-1,1,inclusive = True)]    

# Drop outlier on ls_tot_ta
## First check whether it is still there
df.ls_tot_ta.describe() # yes
df = df[df.ls_tot_ta != df.ls_tot_ta.max()]

#----------------------------------------------
# New box plots
#----------------------------------------------

for var in vars_needed:
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(dict_var_names[var]))
    
    data = df[var]
    ax.boxplot(data)
    
    plt.xticks([1], ['Full Sample (Clean)'])
    
    fig.savefig('Figures\Plots_datacleaning\Scatter_{}_cleaned.png'.format(var))
    plt.clf()
    
''' NOTE: No strangeties -> save to csv
    '''
#----------------------------------------------
# New box plots
#----------------------------------------------
    
color = sns.color_palette("cubehelix", 2)

for var in vars_needed:
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('{}'.format(dict_var_names[var]))
    
    data = df[var]
    plt.subplot(211)
    plt.hist(data, bins = 25, label = 'Not transformed', alpha = .5, color = color[0])
    plt.legend()
    
    plt.subplot(212)
    plt.hist(np.log(data + 1), bins = 25, label = 'Log transformed', alpha = .5, color = color[1]) 
    plt.legend()
    
    fig.savefig('Figures\Plots_datacleaning\Hist_{}.png'.format(var))
    plt.clf()
    
'''NOTE: The data looks better when log transformed --> comes closer to a 
    log-normal distribution '''
    
#------------------------------------------
## Save df
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
df.to_csv('Data\df_wp1_clean.csv')