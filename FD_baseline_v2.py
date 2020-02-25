#------------------------------------------
# OLS for first working paper
# Mark van der Plaat
# January 2020

# Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

# Import method for OLS
from linearmodels import PanelOLS

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

# Import a method to make nice tables
from Code_docs.help_functions.summary3 import summary_col

# Used for the partial R2s
from scipy import stats

#--------------------------------------------
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

# Set the righthand side of the formulas used in analysesFDIV
righthand_x = r'RC7204 + loanratio + roa_a + depratio + comloanratio + mortratio + consloanratio + loanhhi + costinc + RC2170 + bhc'
vars_endo = ['dum_ls','ls_tot_ta'] 
vars_z = ['RIAD4150 + perc_limited_branch'] # In list in case multiple instruments are needed to be run
dep_vars = ['net_coffratio_tot_ta','allowratio_tot_ta','rwata']

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
vars_needed = dep_vars + righthand_x.split(' + ') + vars_endo + list(np.unique(vars_z[0].split(' + ')))
df_full = df[vars_needed]

#---------------------------------------------------
# Setup the data

## Correct dummy and percentage variables for log
df_full['bhc'] = np.exp(df_full.bhc) - 1

## Take logs of the df
df_full = df_full.transform(lambda df: np.log(1 + df))

## Take the first differences
df_full_fd = df_full.groupby(df_full.index.get_level_values(0)).diff(periods = 1).dropna()

## Add time dummies
dummy_full_fd = pd.get_dummies(df_full_fd.index.get_level_values(1))

### Add dummies to the dfs
col_dummy = ['dum' + dummy for dummy in dummy_full_fd.columns.astype(str).str[:4].tolist()]
dummy_full_fd = pd.DataFrame(np.array(dummy_full_fd), index = df_full_fd.index, columns = col_dummy)
df_full_fd[col_dummy] = dummy_full_fd

# Subset the df take the crisis subsets
''' Crisis dates are (based on the NBER recession dates):
    Post-crisis: 2010-2018
    
    Note that the Dodd-Frank act enactment year equals the year the post
    crisis sample starts
    '''

df_post_fd = df_full_fd[df_full_fd.index.get_level_values(1) >= pd.Timestamp(2010,12,31)]
df_predodd_fd = df_full_fd[df_full_fd.index.get_level_values(1) < pd.Timestamp(2010,12,31)]

#---------------------------------------------------
# Load the necessary test functions
def adjR2(r2, n, k): 
    ''' Calculates the adj. R2'''
    adj_r2 = 1 - (1 - r2) * ((n - 1)/(n - (k + 1)))
    return(round(adj_r2, 3))

vecAdjR2 = np.vectorize(adjR2)            
'''-----------------------------------------''' 
#----------------------------------------------
# Setup the method that does the FDIV plus tests
#----------------------------------------------
'''-----------------------------------------''' 

def analysesFD(df, var_ls, righthand_x, time_dummies):
    ''' Performs a FDIV linear model. The second stage takes for dependent variables:
        1) Charge-off rates, 2) Loan Loss allowances, 3) RWA/TA 4) Loan loss provisions.
        The method also correct for unobserved heterogeneity (see Wooldridge p.). Only 
        allows for one endogenous regressor: var_ls
        
        This method does one df and endogenous regressor.'''
        
    # Prelims
    ## Setup the x and z variable vectors
    vars_x = righthand_x.split(' + ')
     
    ## Vector of dependent variables
    dep_var_step2 = ['net_coffratio_tot_ta','allowratio_tot_ta','rwata']
    
    ## Make a string of the time dummy vector
    time_dummies = ' + '.join(time_dummies)
  
    #----------------------------------------------    
    # First check the data on column rank
    ## If not full column rank return empty lists
    rank_full = np.linalg.matrix_rank(df[vars_x+ time_dummies.split(' + ')]) 
    
    if rank_full != len(vars_x + time_dummies.split(' + ')):
        return([],[],[],[])
      
  
    #----------------------------------------------
    # Estimation
    #----------------------------------------------
    ''' In the second stage we regress all dependent variables respectively on
        the fitted values of step 1 the x variables and the time dummies. In step
        2b we include a correction for unobserved heteroskedasticity.
        '''
    res_ols = []
    
    for dep_var in dep_vars:
        res = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                     ' + ' + var_ls + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
        res_ols.append(res) # append to results list
             

    return(res_ols)
           
#----------------------------------------------
#----------------------------------------------
# Analyses
#----------------------------------------------
#----------------------------------------------

# Setup up th rest of the data for loops
## Note: To loop over the four dataframes, we put them in a list
list_dfs = [df_full_fd, df_predodd_fd, df_post_fd]

#----------------------------------------------
# Run models
#----------------------------------------------
# Setup the lists that stores the results from analysesFDIV
res_ols = []

for i in range(len(vars_endo)):
    for data in list_dfs:
        # First set the time dummies (depends on the subset which ones to take)
        time_dummies = ['dum{}'.format(year) for year in data.index.get_level_values(1).unique().astype(str).str[:4][1:].tolist()]
                     
        # Run the model
        res_ols_load = analysesFD(data, vars_endo[i], righthand_x,  time_dummies)
        
        # Save the models
        res_ols.append(res_ols_load)

#----------------------------------------------
# Set things up for nice tables
#----------------------------------------------

# Prelims
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

## Determine regression order
reg_order_ols = vars_endo + righthand_x.split(' + ')

#----------------------------------------------
res_ols_dumls, res_ols_lstotta = split_list(res_ols)

res_ols_dumls_shuffled = []
res_ols_lstotta_shuffled = []

for i in range(len(res_ols_dumls[0])):
    for d in range(len(list_dfs)):
        res_ols_dumls_shuffled.append(res_ols_dumls[d][i])
        res_ols_lstotta_shuffled.append(res_ols_lstotta[d][i])

sum_ols_dumls = summary_col(res_ols_dumls_shuffled, show = 'se', regressor_order = reg_order_ols)
sum_ols_lstotta = summary_col(res_ols_lstotta_shuffled, show = 'se', regressor_order = reg_order_ols)

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
                 'RC7204':'Regulatory Leverage Ratio',
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
                 'nim':'Net Interst Margin',
                 'nnim':'Net Non-Interest Margin',
                 'mortratio':'Mortgage Ratio',
                 'consloanratio':'Consumer Loan Ratio',
                 'agriloanratio':'Agri Loan Ratio',
                 'loanhhi':'Loan HHI',
                 'No. Observations:':'N',
                 'R-squared:':'$R^2$',
                 '$Adj. R^2$':'$Adj. R^2$',
                 'F-test Weak Instruments':'F-test Weak Instruments',
                 'DWH-test':'DWH-test',
                 'P-val Sargan-test':'P-val Sargan-test',
                 'roa_alt_hat':'$\hat{ROA}_{alt}$',
                 'roa_tilde':'$\tilde{ROA}$',
                 'costinc':'Cost-to-income',
                 'roa_a':'$ROA_a$',
                 'Time Dummies':'Time Dummies'}

columns = ['Full','Pre-Dodd-Frank','Post-Dodd-Frank']

#----------------------------------------------
# Make table 
#----------------------------------------------

## Calculate the adj. R2
k_step2 = [mod[0].model.exog.shape[0] for mod in res_ols_dumls] * len(dep_vars)

adjrs_step2_dumls = vecAdjR2(list(sum_ols_dumls.tables[2].iloc[2,:].astype(float)),\
                       list(sum_ols_dumls.tables[2].iloc[1,:].astype(float)), k_step2)
adjrs_step2_lstotta = vecAdjR2(list(sum_ols_dumls.tables[2].iloc[2,:].astype(float)),\
                       list(sum_ols_dumls.tables[2].iloc[1,:].astype(float)), k_step2)

## Change the lower part of the table
step2_lower_table_dumls = sum_ols_dumls.tables[2].iloc[[1],:]
step2_lower_table_dumls.loc[-1] = adjrs_step2_dumls
step2_lower_table_dumls.rename({-1:'$Adj. R^2$'}, axis = 'index', inplace = True)

step2_lower_table_lstotta = sum_ols_lstotta.tables[2].iloc[[1],:]
step2_lower_table_lstotta.loc[-1] = adjrs_step2_lstotta
step2_lower_table_lstotta.rename({-1:'$Adj. R^2$'}, axis = 'index', inplace = True)

### Add to the table
sum_ols_dumls.tables[2] = step2_lower_table_dumls
sum_ols_lstotta.tables[2] = step2_lower_table_lstotta

## Remove the time dummies
step2_param_table_dumls = sum_ols_dumls.tables[1].iloc[:-np.sum((sum_ols_dumls.tables[1].index.str.contains('dum') * 1))*2,:]
step2_param_table_dumls.loc[-1] = ['Yes'] * len(columns) * len(dep_vars)
step2_param_table_dumls.rename({-1:'Time Dummies'}, axis = 'index', inplace = True)

step2_param_table_lstotta = sum_ols_lstotta.tables[1].iloc[:-np.sum((sum_ols_lstotta.tables[1].index.str.contains('dum') * 1))*2,:]
step2_param_table_lstotta.loc[-1] = ['Yes'] * len(columns) * len(dep_vars)
step2_param_table_lstotta.rename({-1:'Time Dummies'}, axis = 'index', inplace = True)

### add to the table
sum_ols_dumls.tables[1] = step2_param_table_dumls
sum_ols_lstotta.tables[1] = step2_param_table_lstotta

## Make new table
table_ols_dumls = pd.concat(sum_ols_dumls.tables[1:3])
table_ols_lstotta = pd.concat(sum_ols_lstotta.tables[1:3])

### Change the columns of the table
columns_ols = [('Loan Charge-offs','Full'),('Loan Charge-offs','Pre-Dodd-Frank'),('Loan Charge-offs','Post-Dodd-Frank'),\
                 ('Loan Loss Allowances','Full'),('Loan Loss Allowances','Pre-Dodd-Frank'),('Loan Loss Allowances','Post-Dodd-Frank'),\
                 ('RWA/TA','Full'),('RWA/TA','Pre-Dodd-Frank'),('RWA/TA','Post-Dodd-Frank')]  
table_ols_dumls.columns = pd.MultiIndex.from_tuples(columns_ols)
table_ols_lstotta.columns = pd.MultiIndex.from_tuples(columns_ols)

### Change Index
table_ols_dumls.index = [dict_var_names[key] for key in table_ols_dumls.index]
table_ols_lstotta.index = [dict_var_names[key] for key in table_ols_lstotta.index]
   
#----------------------------------------------
# Save the tables 
#----------------------------------------------

# Save the tables
path = r'Results\FD_baseline_v2.xlsx'


with pd.ExcelWriter(path) as writer:
    # Save dumls
    table_ols_dumls.to_excel(writer, sheet_name = 'Step 2 Dumls')
    
    # Save LS_tot_ta
   
    table_ols_lstotta.to_excel(writer, sheet_name = 'Step 2 LSTA')