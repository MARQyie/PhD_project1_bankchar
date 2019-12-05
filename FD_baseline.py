#------------------------------------------
# FD Baseline for first working paper
# Mark van der Plaat
# October 2019 

 # Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

# Import method for POLS (also does FE)
from linearmodels import PanelOLS

import sys # to use the help functions needed
sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

from summary3 import summary_col

from linearmodels.utility import WaldTestStatistic
#--------------------------------------------
# Set parameters 
log = True # If set to False the program estimates the model without logs and with size

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
if log:
    df['dum_ls'] = np.exp((df.ls_tot > 0) * 1) - 1 #will be taken the log of later
else:
    df['dum_ls'] = (df.ls_tot > 0) * 1    

## Take a subset of variables (only the ones needed)
vars_needed = ['distance','provratio','rwata','net_coffratio_tot_ta',\
               'allowratio_tot_ta','ls_tot_ta','dum_ls','size',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170',\
               'num_branch', 'bhc', 'RIAD4150', 'perc_limited_branch',\
               'perc_full_branch', 'unique_states','UNIT']
df_full = df[vars_needed]

#---------------------------------------------------
# Setup the data

## Correct dummy and percentage variables for log
if log:
    df_full['bhc'] = np.exp(df_full.bhc) - 1

## Take logs of the df
if log:
    df_full = df_full.transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)

## Add the x_xbar to the df
if log:
    x_xbar = df_full[['RC7205','loanratio','roa',\
                      'depratio','comloanratio','RC2170','bhc']].transform(lambda df: df - df.mean())
    df_full[[x + '_xbar' for x in ['RC7205','loanratio',\
                                   'roa','depratio','comloanratio','RC2170','bhc']]] = x_xbar
else:
    x_xbar = df_full[['RC7205','loanratio','roa',\
                          'depratio','comloanratio','size','bhc']].transform(lambda df: df - df.mean())
    df_full[[x + '_xbar' for x in ['RC7205','loanratio',\
                                   'roa','depratio','comloanratio','size','bhc']]] = x_xbar

## Take the first differences
df_full_fd = df_full.groupby(df_full.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

## Add time dummies
dummy_full_fd = pd.get_dummies(df_full_fd.index.get_level_values(1))

### Add dummies to the dfs
col_dummy = ['dum' + dummy for dummy in dummy_full_fd.columns.astype(str).str[:4].tolist()]
dummy_full_fd = pd.DataFrame(np.array(dummy_full_fd), index = df_full_fd.index, columns = col_dummy)
df_full_fd[col_dummy] = dummy_full_fd

# Subset the df take the crisis subsets
''' Crisis dates are (based on the NBER recession dates):
    Pre-crisis: 2001-2006
    Crisis: 2007-2009
    Post-crisis: 2010-2018
    
    Note that the Dodd-Frank act enactment year equals the year the post
    crisis sample starts
    '''

df_pre_fd = df_full_fd[df_full_fd.index.get_level_values(1) <= pd.Timestamp(2006,12,31)]
df_during_fd = df_full_fd[(df_full_fd.index.get_level_values(1) > pd.Timestamp(2006,12,31)) & (df_full_fd.index.get_level_values(1) < pd.Timestamp(2010,12,31))]
df_post_fd = df_full_fd[df_full_fd.index.get_level_values(1) >= pd.Timestamp(2010,12,31)]
df_predodd_fd = df_full_fd[df_full_fd.index.get_level_values(1) < pd.Timestamp(2010,12,31)]

'''-----------------------------------------''' 
#----------------------------------------------
# Setup the method that does the FD plus tests
#----------------------------------------------
'''-----------------------------------------''' 

def analysesFD(df, var_ls, righthand_x, time_dummies):
        
    # Prelims
    ## Setup the x variable vector
    vars_x = righthand_x.split(' + ')
    
    ## Vector of dependent variables

    dep_vars = ['net_coffratio_tot_ta','allowratio_tot_ta','provratio']
    
    ## Make a string of the time dummy vector
    time_dummies = ' + '.join(time_dummies)
    
    ## Determine regression order
    reg_order = [var_ls] + vars_x
    
    #----------------------------------------------    
    # First check the data on column rank
    ## If not full column rank return empty lists
    rank_full = np.linalg.matrix_rank(df[vars_x + time_dummies.split(' + ')]) 
    
    if rank_full != len(vars_x + time_dummies.split(' + ')):
        return([],[],[])
    
    #----------------------------------------------
    # OLS Estimation
    #----------------------------------------------

    res_ols = []
    
    for dep_var in dep_vars:
        res = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                     ' + ' + var_ls + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
        res_ols.append(res) # append to results list

    # Create the summary of the models
    sum_ols = summary_col(res_ols, show = 'se', regressor_order = reg_order)
 
    #----------------------------------------------
    # Tests
    #----------------------------------------------
    
    #----------------------------------------------                                
    ## RESET
    
    reset_tests = []
    
    for result in res_ols:
        # Set up the variables for the regression
        df['u_hat'] = result.resids
        df['ls2'] = np.power(result.fitted_values, 2)
        df['ls3'] = np.power(result.fitted_values, 3)
        df['ls4'] = np.power(result.fitted_values, 4)
        
        # loop over all three possible test regressions and calculate the pval with a Wald test
        null = 'H0: no misspecification'
        name = 'RESET test for functional for misspecification'
        pvals = []
        
        for i in range(3):
            powers = ' + '.join(['ls{}'.format(j + 2) for j in range(i)])
            reset_reg = PanelOLS.from_formula('u_hat' + ' ~ ' + righthand_x +\
                     ' + ' + powers + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
            
            stat = df.shape[0] * reset_reg.rsquared
            dof = i + 1
            
            pvals.append(WaldTestStatistic(stat, null, dof, name=name).pval)
        
        reset_tests.append(pvals)
        
    if var_ls == 'dum_ls':
        return(sum_ols,reset_tests,0)
    else:
        return(sum_ols,reset_tests,1)

#----------------------------------------------
#----------------------------------------------
# Analyses
#----------------------------------------------
#----------------------------------------------

# Set the righthand side of the formulas
if log:
    righthand_x = r'RC7205 + loanratio + roa + depratio + comloanratio + RC2170'
else:
    righthand_x = r'RC7205 + loanratio + roa + depratio + comloanratio + size'

time_dummies = ' + '.join(col_dummy[1:])
   
vars_ls = ['ls_tot_ta'] 

# Setup up th rest of the data for loops
## Note: To loop over the four dataframes, we put them in a list
list_dfs = list_dfs = [df_full_fd, df_pre_fd, df_during_fd,df_post_fd, df_predodd_fd]

#----------------------------------------------
# Run models
#----------------------------------------------
''' Note: To get the results for all four dfs, two loan sale variables and all
    combinations of instrumental variables, we use a triple loop and save the 
    results to a list. 
    '''

# Setup the lists that stores the results from analysesFDIV
res_ols = []
pvals_reset = []
list_ls_vars = []

for data in list_dfs:
    # First set the time dummies (depends on the subset which ones to take)
    time_dummies = ['dum{}'.format(year) for year in data.index.get_level_values(1).unique().astype(str).str[:4][1:].tolist()]
    
    for i in range(len(vars_ls)):
        res_ols_load, pvals_reset_load, ls_var_load =\
        analysesFD(data, vars_ls[i], righthand_x, time_dummies)
            
        res_ols.append(res_ols_load)
        pvals_reset.append(pvals_reset_load)
        list_ls_vars.append(ls_var_load)

#----------------------------------------------
# Save the tables and scoring vector
#----------------------------------------------
# Make a vector containing names for the sheets

names_dfs = ['Full_','Pre_','During_','Post_','PreDodd_']
names_dfs_endo = [(a + b) for a in names_dfs for b in vars_ls] 

#----------------------------------------------
# Prelims
''' Note: The prelims set up a dictionary to use in making nice tables.   
    '''
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

### Add the ghat_w variables to the dict
vars_x = pd.Series(righthand_x.split(' + ')).unique()

## Add the time dummies to the dict
keys_time_dummies = df_full_fd.columns[df_full_fd.columns.str.contains('dum2')]
values_time_dummies = 'I(t=' + keys_time_dummies.str[3:] + ')'

dict_td = {}

for key, name in zip(keys_time_dummies, values_time_dummies):
    dict_td[key] = name
    
dict_var_names.update(dict_td) 

### Specify a list with variable names to drop when assigning variable labels
list_drop = ['', '\t Results Summary', 'Model:', 'No. Observations:', 'R-squared:', 'F-statistic:', 'Covariance Type:', 'note:', '\t Std. error in parentheses.', '\t * p<.1, ** p<.05, ***p<.01']

#----------------------------------------------
# Save step 1, step 2 and the test tables
if log:
    path = r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char\FD_v2_log.xlsx'
else:
    path = r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char\FD_v2.xlsx'

with pd.ExcelWriter(path) as writer:
    for i in range(len(res_ols)):
        
        # Get the index (variable names) from the summary files
        vars_ols = [x for x in pd.concat(res_ols[i].tables,axis=0).index.tolist() if x not in list_drop]
        
        # Add a dict entry for the correct definition of G_hat
        if list_ls_vars[i] == 0:
            dict_var_names.update({'G_hat_fd':'Dummy Loan Sales'})
        else:
            dict_var_names.update({'G_hat_fd':'Loan Sales/TA'})
        
        # Determine the variable names
        var_names_step1 = [dict_var_names[key] for key in vars_ols]
        
        # Save the variable names and keys in a dict
        dict_names_step1 = dict(zip(vars_ols,var_names_step1))
        
        # Save the results of step 1 and 2 and the test statistics to the excel file
        res_ols[i].to_excel(writer, sheet_name = names_dfs_endo[i],rename_index = dict_names_step1)
