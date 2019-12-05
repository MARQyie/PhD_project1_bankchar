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
sns.set(style = 'whitegrid',font_scale=1.2)

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

from linearmodels import PanelOLS
from linearmodels.panel import compare

import sys # to use the help functions needed
sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

from summary3 import summary_col

# Import method that adds a constant to a df
from statsmodels.tools.tools import add_constant


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


## Dummy variable for loan sales
df['dum_ls'] = np.exp((df.ls_tot > 0) * 1) - 1 #will be taken the log of later 

## Take a subset of variables (only the ones needed)
vars_needed = ['distance','provratio','net_coffratio_tot_ta',\
               'allowratio_tot_ta','ls_tot_ta','dum_ls',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170',\
               'num_branch', 'bhc', 'RIAD4150', 'perc_limited_branch',\
               'unique_states']
df_full = df[vars_needed]

#---------------------------------------------------
# Setup the data

## Correct dummy and percentage variables for log
df_full['bhc'] = np.exp(df_full.bhc) - 1

## Take logs of the df
df_full = df_full.transform(lambda df: np.log(1 + df))

## Take the first differences
df_full_fd = df_full.groupby(df_full.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

# Subset the df take the crisis subsets
df_pre_fd = df_full_fd[df_full_fd.index.get_level_values(1) <= pd.Timestamp(2006,12,31)]
df_during_fd = df_full_fd[(df_full_fd.index.get_level_values(1) > pd.Timestamp(2006,12,31)) & (df_full_fd.index.get_level_values(1) < pd.Timestamp(2010,12,31))]
df_post_fd = df_full_fd[df_full_fd.index.get_level_values(1) >= pd.Timestamp(2010,12,31)]
df_predodd_fd = df_full_fd[df_full_fd.index.get_level_values(1) < pd.Timestamp(2010,12,31)]

#------------------------------------------------
# Create the first difference df
## First select the variables
## Make dict that contains all variables and names
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

labels = [dict_var_names[var] for var in vars_needed]

#-------------------------------------------------
# Correlation Matrices
corr_matrix_full = df_full_fd.corr(method = 'spearman')
corr_matrix_precrisis = df_pre_fd.corr(method = 'spearman')
corr_matrix_crisis = df_during_fd.corr(method = 'spearman')
corr_matrix_postcrisis = df_post_fd.corr(method = 'spearman')
corr_matrix_predodd = df_predodd_fd.corr(method = 'spearman')

# Plot
## Setup everything for loop
corr_matrices = [corr_matrix_full,corr_matrix_precrisis,corr_matrix_crisis,\
                 corr_matrix_postcrisis,corr_matrix_predodd]
titles = ['Correlation Matrix Full Sample','Correlation Matrix Pre-Crisis Sample',\
          'Correlation Matrix Crisis Sample','Correlation Matrix Post-Crisis/Dodd-Frank Sample',\
          'Correlation Matrix Pre-Dodd-Frank Sample']
paths = ['Corr_matrix_full_sample.png','Corr_matrix_precrisis_sample.png',\
         'Corr_matrix_crisis_sample.png','Corr_matrix_postcrisis_sample.png',\
         'Corr_matrix_predodd_sample.png']

## Loop over all plots
for matrix, title, path in zip(corr_matrices, titles, paths):
    fig, ax = plt.subplots(figsize=(20, 16))
    plt.title(title, fontsize=20)
    ax = sns.heatmap(
        matrix, 
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
            
    fig.savefig(path)     
    
    plt.close(fig)
    plt.clf()

        