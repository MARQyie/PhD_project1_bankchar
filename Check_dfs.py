#------------------------------------------
# Check the dfs on outliers etc.
# Mark van der Plaat
# December 2019 

 # Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

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
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

## Make multi index
df.date = pd.to_datetime(df.date.astype(str))
df.set_index(['IDRSSD','date'],inplace=True)

## Drop missings on distance
df.dropna(subset = ['distance'], inplace = True)

## Dummy variable for loan sales
df['dum_ls'] = np.exp((df.ls_tot > 0) * 1) - 1 #will be taken the log of later

## Take a subset of variables (only the ones needed)
vars_needed = ['provratio','rwata','net_coffratio_tot_ta',\
               'allowratio_tot_ta','ls_tot_ta','dum_ls','size',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170',\
               'num_branch', 'bhc', 'RIAD4150', 'perc_limited_branch',\
               'unique_states','distance']
df_full = df[vars_needed]

## drop NaNs
df_full.dropna(subset = ['provratio','rwata','net_coffratio_tot_ta','allowratio_tot_ta',\
               'ls_tot_ta','RC7205','loanratio','roa',\
               'depratio','comloanratio','RC2170','size'], inplace = True)

#---------------------------------------------------
# Setup the data

## Correct dummy and percentage variables for log
df_full['bhc'] = np.exp(df_full.bhc) - 1

## Take logs of the df
df_full = df_full.transform(lambda df: np.log(1 + df))

## Add time dummies
dummy_full = pd.get_dummies(df_full.index.get_level_values(1))

### Add dummies to the dfs
col_dummy = ['dum' + dummy for dummy in dummy_full.columns.astype(str).str[:4].tolist()]
dummy_full = pd.DataFrame(np.array(dummy_full), index = df_full.index, columns = col_dummy)
df_full[col_dummy] = dummy_full

# Subset the df take the crisis subsets
df_pre = df_full[df_full.index.get_level_values(1) <= pd.Timestamp(2006,12,31)]
df_during = df_full[(df_full.index.get_level_values(1) > pd.Timestamp(2006,12,31)) & (df_full.index.get_level_values(1) < pd.Timestamp(2010,12,31))]
df_post = df_full[df_full.index.get_level_values(1) >= pd.Timestamp(2010,12,31)]

#---------------------------------------------------
# Define the y and x variables and set up the list of dfs (for looping)
## NOTE: No dummies

y = ['provratio','net_coffratio_tot_ta','allowratio_tot_ta']
x = ['ls_tot_ta','RC7205','loanratio','roa',\
     'depratio','comloanratio','RC2170','num_branch',\
     'RIAD4150','perc_limited_branch','unique_states','distance']

list_dfs = [df_full, df_pre, df_during,df_post]
df_names = ['Full', 'Pre-crisis', 'Crisis', 'Post-crisis']

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
                 'UNIT':'Unit Bank Indicator'}

#---------------------------------------------- 
#----------------------------------------------
# Scatter plots
#----------------------------------------------
#----------------------------------------------
for df, name in zip(list_dfs, df_names):
    for y_var in y:
        for x_var in x:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.title('{} Sample'.format(name))
            ax.set(xlabel = dict_var_names[x_var], ylabel = dict_var_names[y_var])
            plt.scatter(df[x_var], df[y_var])
            plt.tight_layout()
            
            fig.savefig('Scatter_plots/Scatter_{}_{}_{}.png'.format(x_var,y_var,name))
            plt.clf()

#---------------------------------------------- 
#----------------------------------------------
# Box plots
#----------------------------------------------
#----------------------------------------------

for var in y + x:
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(dict_var_names[var]))
    
    data = [list_dfs[x][var] for x in range(len(list_dfs))]
    ax.boxplot(data)
    
    plt.xticks([1, 2, 3, 4], df_names)
    
    fig.savefig('Box_plots/Scatter_{}.png'.format(var))
    plt.clf()
    
#---------------------------------------------- 
#----------------------------------------------
# Plots in First differences
#----------------------------------------------
#----------------------------------------------
  
# Prepare the data
## Take the first differences
df_full_fd = df_full.groupby(df_full.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

## Subset the df take the crisis subsets
df_pre_fd = df_full_fd[df_full_fd.index.get_level_values(1) <= pd.Timestamp(2006,12,31)]
df_during_fd = df_full_fd[(df_full_fd.index.get_level_values(1) > pd.Timestamp(2006,12,31)) & (df_full_fd.index.get_level_values(1) < pd.Timestamp(2010,12,31))]
df_post_fd = df_full_fd[df_full_fd.index.get_level_values(1) >= pd.Timestamp(2010,12,31)]

# List dfs fd
list_dfs_fd = [df_full_fd, df_pre_fd, df_during_fd, df_post_fd]

#----------------------------------------------
# Box plots
#----------------------------------------------

for var in y + x:
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{} (First Differences)'.format(dict_var_names[var]))
    
    data = [list_dfs_fd[x][var] for x in range(len(list_dfs_fd))]
    ax.boxplot(data)
    
    plt.xticks([1, 2, 3, 4], df_names)
    
    fig.savefig('Box_plots/Scatter_{}_fd.png'.format(var))
    plt.clf()