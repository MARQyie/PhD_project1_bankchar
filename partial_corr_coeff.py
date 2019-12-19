# Partial correlation coefficients
 # Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

import numpy as np
from scipy import stats, linalg

#--------------------------------------------
''' This script estimates the treatment effect of loan sales on credit risk
    with an IV estimation procedure. The procedure has two steps
    
    Step 1: Estimate a linear model of LS on X and Z, where X are the exogenous
    variables and Z are the instruments. Obtain the fitted probabilities G()
    
    Step 2: Do a OLS of CR on 1, G_hat, X, G_hat(X-X_bar)
    
    Both steps are in first differences and include time dummies to account for
    any time fixed effects.
    
    Dynamic effects are not included.
    '''   

''' Inputs and outputs of the script

    INPUT: 
        A pandas dataframe with Call Report data and data from the summary
        of deposits
        
        A toggle switch to either use logs or not to estimate the system
    
    OUTPUT: 
        Excel file with the results of the first and second step and the test
        results of a T-test/F-test for weak instruments, a test for endogeneity
        of the Loan Sales variable and a Sargan tests for overidentifying restrictions
        
        A txt-file with a scoring vector containing a score between 0 and 1 
        that scores how good the instrumental variables are working
    '''

#---------------------------------------------- 
#----------------------------------------------
# Prelims
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# Load data and add needed variables

# Load df
df = pd.read_csv('Data\df_wp1_clean.csv', index_col = 0)

## Make multi index
df.date = pd.to_datetime(df.date.astype(str))
df.set_index(['IDRSSD','date'],inplace=True)


## Dummy variable for loan sales
df['dum_ls'] = np.exp((df.ls_tot > 0) * 1) - 1 #will be taken the log of later

## Take a subset of variables (only the ones needed)
vars_needed = ['distance','provratio','net_coffratio_tot_ta',\
               'allowratio_tot_ta','ls_tot_ta','dum_ls','size',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170',\
               'num_branch', 'bhc', 'RIAD4150', 'perc_limited_branch',\
               'unique_states','mortratio','consloanratio',\
               'agriloanratio','loanhhi','nim','nnim','roa_alt']
vars_model = ['net_coffratio_tot_ta','allowratio_tot_ta','provratio','dum_ls','ls_tot_ta',\
               'RC7205','loanratio','roa','roa_alt','nim','nnim','depratio','comloanratio','RC2170','bhc',\
               'mortratio','consloanratio','loanhhi','RIAD4150', 'perc_limited_branch'] 

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

#----------------------------------------------
# Method
#----------------------------------------------

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr

#----------------------------------------------
# Calculate the matrices
#----------------------------------------------
    
dict_var_names = {'':'',
                  'distance':'Max Distance Branches',
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
                 'roa_alt':'ROA Alt.',
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
                 'nim':'Net Interst Margin',
                 'nnim':'Net Non-Interest Margin',
                 'mortratio':'Mortgage Ratio',
                 'consloanratio':'Consumer Loan Ratio',
                 'agriloanratio':'Agri Loan Ratio',
                 'loanhhi':'Loan HHI',
                 'No. Observations:':'N',
                 'R-squared:':'$R^2$',
                 'F-test Weak Instruments':'F-test Weak Instruments',
                 'DWH-test':'DWH-test',
                 'P-val Sargan-test':'P-val Sargan-test'}

labels = [dict_var_names[var] for var in vars_model]
    
partial_corr_full = pd.DataFrame(partial_corr(df_full_fd[vars_model]), columns = labels, index = labels)