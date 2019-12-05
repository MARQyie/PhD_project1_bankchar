#------------------------------------------
# Correlation matrices for the first working paper
# Mark van der Plaat
# October 2019 

   
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

from linearmodels import PanelOLS
from linearmodels.panel import compare

import sys # to use the help functions needed
sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

from summary3 import summary_col

# Import method that adds a constant to a df
from statsmodels.tools.tools import add_constant

# Set parameters 
change_ls = False # If set to False the program will run a different subset and append it to the excel

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip())
df.set_index(['IDRSSD','date'],inplace=True)

# Drop missings on distance
df.dropna(subset = ['distance'], inplace = True)

# Dummy variable for loan sales
df['dum_ls'] = np.exp((df.ls_tot > 0) * 1) - 1

# Subset the df
if change_ls:
    intersect = np.intersect1d(df[df.ls_tot_ta > 0].index.\
                               get_level_values(0).unique(),\
                               df[df.ls_tot_ta == 0].index.\
                               get_level_values(0).unique())
    df_ls = df[df.index.get_level_values(0).isin(intersect)]
else:
    ## Kick out the community banks (based on Stiroh, 2004)  
    ids_comm = df[((df.index.get_level_values(1) == pd.Timestamp(2018,12,30)) &\
                     (df.RC2170 < 3e5) & (df.bhc == 0))].index.get_level_values(0).unique().tolist() 
    ids_tot = df.index.get_level_values(0).unique().tolist()
    ids_sub = [x for x in ids_tot if x not in ids_comm]
    df_ls = df[df.index.get_level_values(0).isin(ids_sub)]   
  

## Take logs
df = df.select_dtypes(include = ['float64','int','int64']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)
df_ls = df_ls.select_dtypes(include = ['float64','int','int64']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)

## Drop NaNs on subset
df.dropna(subset = ['rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)
df_ls.dropna(subset = ['rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)

#------------------------------------------------
# Create the first difference df
## First select the variables
vars_list = ['rwata','net_coffratio_tot_ta','allowratio_tot_ta','provratio','ls_tot_ta',\
             'dum_ls','RC7205','loanratio','roa','nim','depratio','comloanratio','RC2170',\
             'num_branch', 'bhc', 'RIAD4150', 'distance', 'perc_limited_branch','perc_full_branch',\
             'unique_states','UNIT']
labels = ['RWATA', 'Charge-offs-to-TA','Allowances-to-TA', 'Provision Ratio', 'Loan-Sales-to-TA',\
          'Dummy Loan Sales','Regulatory Capital Ratio', 'Loans-to-TA','Return on Assets',\
          'Net Interest Margin','Deposit Ratio','Commercial Loan Ratio','Total Assets',\
          'Number of Branches', 'BHC Indicator', 'Number of Employees', 'Max. Distance Branches',\
          'Limited Branches (in %)','Full Branches (in %)','Number of States Active','Unit Indicator']

## Transform the data
df_fd = df[vars_list].groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
df_ls_fd = df_ls[vars_list].groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

#-------------------------------------------------
# Correlation Matrices
corr_matrix_full = df_fd.corr(method = 'spearman')
corr_matrix_subset = df_ls_fd.corr(method = 'spearman')

# Plot
## Full sample
fig, ax = plt.subplots(figsize=(20, 16))
plt.title('Correlation Matrix Full Sample', fontsize=20)
ax = sns.heatmap(
    corr_matrix_full, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    annot = True
)
ax.set_xticklabels(
    labels,
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(
    labels,
);
        
fig.savefig('Corr_matrix_1_full_sample.png')     

plt.close(fig)

## Subset
fig, ax = plt.subplots(figsize=(20, 16))
if change_ls:
    plt.title('Correlation Matrix Change in Loan Sales', fontsize=20)
else:
    plt.title('Correlation Matrix No Community Banks', fontsize=20)
ax = sns.heatmap(
    corr_matrix_subset, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    annot=True
)
ax.set_xticklabels(
    labels,
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(
    labels,
);

if change_ls:
    fig.savefig('Corr_matrix_2a_change_ls.png')  
else:
    fig.savefig('Corr_matrix_2b_no_comm_banks.png')  

plt.close(fig)
  
## Not FD
corr_matrix_full_nofd = df[vars_list].corr(method = 'spearman')
corr_matrix_subset_nofd = df_ls[vars_list].corr(method = 'spearman')
        
ax = sns.heatmap(
    corr_matrix_full_nofd, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
     
        