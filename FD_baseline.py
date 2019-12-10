#------------------------------------------
# FD Baseline for first working paper
# Mark van der Plaat
# October 2019 

 # Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

# Import method for POLS (also does FE)
from linearmodels import PanelOLS

from Code_docs.help_functions.summary3 import summary_col

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
df = pd.read_csv('Data\df_wp1_clean.csv', index_col = 0)

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
               'perc_full_branch', 'unique_states','UNIT','nim']
df_full = df[vars_needed]

#---------------------------------------------------
# Setup the data

## Correct dummy and percentage variables for log
if log:
    df_full['bhc'] = np.exp(df_full.bhc) - 1

## Take logs of the df
if log:
    df_full = df_full.transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)

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
        return(res_ols,reset_tests,0)
    else:
        return(res_ols,reset_tests,1)

#----------------------------------------------
#----------------------------------------------
# Analyses
#----------------------------------------------
#----------------------------------------------

# Set the righthand side of the formulas
if log:
    righthand_x = r'RC7205 + loanratio + nim + depratio + comloanratio + RC2170 + bhc'
else:
    righthand_x = r'RC7205 + loanratio + nim + depratio + comloanratio + size + bhc'

time_dummies = ' + '.join(col_dummy[1:])
   
vars_ls = ['dum_ls','ls_tot_ta'] 

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

for i in range(len(vars_ls)):
    for data in list_dfs:
    # First set the time dummies (depends on the subset which ones to take)
        time_dummies = ['dum{}'.format(year) for year in data.index.get_level_values(1).unique().astype(str).str[:4][1:].tolist()]
    
        res_ols_load, pvals_reset_load, ls_var_load =\
        analysesFD(data, vars_ls[i], righthand_x, time_dummies)
            
        res_ols.append(res_ols_load)
        pvals_reset.append(pvals_reset_load)
        list_ls_vars.append(ls_var_load)

#----------------------------------------------
# Set things up for nice tables
#----------------------------------------------

# Prelims
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

## Determine regression order
reg_order = vars_ls + righthand_x.split(' + ')

#----------------------------------------------
res_dumls, res_lstotta = split_list(res_ols) 

res_dumls_shuffled = []
res_lstotta_shuffled = []
first_loop = True

for d in range(len(list_dfs)):
    for i in range(len(res_dumls[d])):
        if first_loop:
            res_dumls_shuffled.append([res_dumls[d][i]])
            res_lstotta_shuffled.append([res_lstotta[d][i]])
        else:
            res_dumls_shuffled[i].append(res_dumls[d][i])
            res_lstotta_shuffled[i].append(res_lstotta[d][i])
    first_loop = False

sum_dumls = [] 
sum_lstotta = [] 
for vec_dum, vec_ls in zip(res_dumls_shuffled,res_lstotta_shuffled):
    sum_dumls.append(summary_col(vec_dum, show = 'se', regressor_order = reg_order))    
    sum_lstotta.append(summary_col(vec_ls, show = 'se', regressor_order = reg_order)) 

#----------------------------------------------
# Prelims for making nice tables
#----------------------------------------------
## Make dict that contains all variables and names
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
                 'nim':'Net Interest Margin',
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
                 'No. Observations:':'N',
                 'R-squared:':'$R^2$',
                 'F-test Weak Instruments':'F-test Weak Instruments',
                 'DWH-test':'DWH-test',
                 'P-val Sargan-test':'P-val Sargan-test'}

## Add the time dummies to the dict
keys_time_dummies = df_full_fd.columns[df_full_fd.columns.str.contains('dum2')]
values_time_dummies = 'I(t=' + keys_time_dummies.str[3:] + ')'

dict_td = {}

for key, name in zip(keys_time_dummies, values_time_dummies):
    dict_td[key] = name
    
dict_var_names.update(dict_td) 

columns = ['Full','Pre-Crisis','Crisis','Pre-Dodd-Frank','Post-Crisis/Dodd-Frank']

#----------------------------------------------
# Make table 
#----------------------------------------------

list_tables_dumls = []
list_tables_lstotta = []

for i in range(len(sum_dumls)):

    ## Change the lower part of the table
    lower_table_dumls = sum_dumls[i].tables[2].iloc[[1,2],:]
    lower_table_lstotta = sum_lstotta[i].tables[2].iloc[[1,2],:]
       
    ### Add to the table
    sum_dumls[i].tables[2] = lower_table_dumls
    sum_lstotta[i].tables[2] = lower_table_lstotta
    
    ## Make new table
    table_dumls = pd.concat(sum_dumls[i].tables[1:3])
    table_lstotta = pd.concat(sum_lstotta[i].tables[1:3])
    
    ### Change the columns of the table
    table_dumls.columns = columns
    table_lstotta.columns = columns
    
    ### Change the index
    dict_var_names.update({'G_hat_fd':'Dummy Loan Sales'})
    table_dumls.index = [dict_var_names[key] for key in table_dumls.index]
    dict_var_names.update({'G_hat_fd':'Loan Sales/TA'})
    table_lstotta.index = [dict_var_names[key] for key in table_lstotta.index]
        
    ## Append to the list
    list_tables_dumls.append(table_dumls)
    list_tables_lstotta.append(table_lstotta)

#----------------------------------------------
# Save the tables 
#----------------------------------------------
# Prelims
sheets = ['Charge-offs','Allowance','Provisions']

# Save the tables
path = r'Results\FD_v2_log.xlsx'

with pd.ExcelWriter(path) as writer:
    # Save dumls
    for i in range(len(list_tables_dumls)):
        list_tables_dumls[i].to_excel(writer, sheet_name = 'Dumls - {}'.format(sheets[i])) 
    
    # Save LS_tot_ta
    for i in range(len(list_tables_lstotta)):
        list_tables_lstotta[i].to_excel(writer, sheet_name = 'LSTA - {}'.format(sheets[i]))

