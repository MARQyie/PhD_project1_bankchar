#------------------------------------------
# IV treatment model for first working paper
# Mark van der Plaat
# January 2020

# Import packages
import pandas as pd
import numpy as np
import itertools
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

# Import method for OLS
from linearmodels import PanelOLS

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

# Import a method to make nice tables
from Code_docs.help_functions.summary3 import summary_col

'''OLD
# Used for the partial R2s
from scipy import stats
'''

#--------------------------------------------
''' This script estimates the treatment effect of loan sales on credit risk
    with an IV estimation procedure. The procedure has two steps
    
    Step 1: Estimate a linear model of LS on X and Z, where X are the exogenous
    variables and Z are the instruments. Obtain the fitted probabilities G()
    
    Step 2: Do a OLS of CR on 1, G_hat, X
    
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

# Set the righthand side of the formulas used in analysesFDIV
righthand_x = r'loanratio + depratio + comloanratio + mortratio + consloanratio + loanhhi + costinc + RC2170 + bhc'
vars_endo = ['dum_ls','ls_tot_ta'] 
vars_z = ['RIAD4150 + perc_limited_branch'] # In list in case multiple instruments are needed to be run
dep_vars = ['net_coffratio_tot_ta','allowratio_tot_ta','rwata']
num_cores = mp.cpu_count()

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

def fTestWeakInstruments(y, fitted_full, fitted_reduced, dof = 4):
    ''' Simple F-test to test the strength of instrumental variables. See
        Staiger and Stock (1997)
        
        y : True y values
        fitted_full : fitted values of the first stage with instruments
        fitted_reduced : fitted values first stage without instruments
        dof : number of instruments'''
    
    # Calculate the SSE and MSE
    sse_full = np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    sse_reduced =  np.sum([(y.values[i] - fitted_reduced.values[i][0])**2 for i in range(y.shape[0])])
    
    mse_full = (1 / y.shape[0]) * np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    
    # Calculate the statistic
    f_stat = ((sse_reduced - sse_full)/dof) / mse_full
    
    return f_stat

def sargan(resids, x, z, nendog = 1):
    '''Function performs a sargan test (no heteroskedasity) to check
        the validity of the overidentification restrictions. H0: 
        overidentifying restrictions hold.
        
        resids : residuals of the second stage
        x : exogenous variables
        z : instruments 
        nendog : number of endogenous variables'''  

    nobs, ninstr = z.shape
    name = 'Sargan\'s test of overidentification'

    eps = resids.values[:,None]
    u = annihilate(eps, pd.concat([x,z],axis = 1))
    stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
    null = 'The overidentification restrictions are valid'

    return WaldTestStatistic(stat, null, ninstr - nendog, name=name)

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

def analysesFDIV(var_ls, df, righthand_z, righthand_x, time_dummies):
    ''' Performs a FDIV linear model. The second stage takes for dependent variables:
        1) Charge-off rates, 2) Loan Loss allowances, 3) RWA/TA 4) Loan loss provisions.
        The method also correct for unobserved heterogeneity (see Wooldridge p.). Only 
        allows for one endogenous regressor: var_ls
        
        This method does one df and endogenous regressor.'''
        
    # Prelims
    ## Setup the x and z variable vectors
    vars_x = righthand_x.split(' + ')
    vars_z = righthand_z.split(' + ')
    
    ## Number of z variables
    num_z = righthand_z.count('+') + 1
    
    ## Vector of dependent variables
    dep_var_step2 = ['net_coffratio_tot_ta','allowratio_tot_ta','rwata']
    
    ## Make a string of the time dummy vector
    time_dummies = ' + '.join(time_dummies)
  
    #----------------------------------------------    
    # First check the data on column rank
    ## If not full column rank return empty lists
    rank_full = np.linalg.matrix_rank(df[vars_x + vars_z + time_dummies.split(' + ')]) 
    
    if rank_full != len(vars_x + vars_z + time_dummies.split(' + ')):
        return([],[],[],[],[])
      
    #----------------------------------------------
    # STEP 1: First Stage
    #----------------------------------------------
    ''' In the first stage we regress the loan sales variable on the x and z variables
        and the time dummies and calculate the fitted values "G_hat_fd". 
        
        The summary of the first step is returned
        '''  
                          
    # Estimate G_hat
    res_step1 = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + righthand_z + ' + ' + time_dummies,\
                                      data = df).fit(cov_type = 'clustered', cluster_entity = True)
    
    df['G_hat_fd'] = res_step1.fitted_values
    
    #----------------------------------------------
    # Step 2: Second Stage
    #----------------------------------------------
    ''' In the second stage we regress all dependent variables respectively on
        the fitted values of step 1 the x variables and the time dummies. In step
        2b we include a correction for unobserved heteroskedasticity.
        '''
        
    res_step2 = []
    
    for dep_var in dep_var_step2:
        # without correction unobserved heterogeneity
        res_step2a = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                     ' + ' + 'G_hat_fd' + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
        res_step2.append(res_step2a) # append to results list
             
    #----------------------------------------------
    # Tests
    '''We test for three things:
        1) The strength of the instrument using a Staiger & Stock type test.\
           F-stat must be > 10.
        2) A DWH test whether dum_ls or ls_tot_ta is endogenous. H0: variable is exogenous
        3) A Sargan test to test the overidentifying restrictions. H0: 
           overidentifying restrictions hold
        
        For only one instrument (num_z == 1) we use the p-val of the instrument in step 1) 
            and we do not do a Sargan test'''        
    #----------------------------------------------
    
    #----------------------------------------------                                
    ## Weak instruments
    
    if num_z == 1:
        f_test_step1 = res_step1.pvalues[vars_z].values
    else:
        res_step1b = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + time_dummies, data = df).fit(cov_type = 'clustered', cluster_entity = True)
        f_test_step1 = fTestWeakInstruments(df[var_ls], res_step1.fitted_values, res_step1b.fitted_values, num_z)
    
    #----------------------------------------------
    ## Endogenous loan sales variable
    
    df['resid_step1'] = res_step1.resids

    pvals_ls_endo = []
    for dep_var in dep_var_step2:
        res_endo = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                       ' + ' + var_ls + ' + ' + 'resid_step1' + ' + ' + time_dummies, data = df).\
                       fit(cov_type = 'clustered', cluster_entity = True)
        pvals_ls_endo.append(res_endo.pvalues['resid_step1'])
              
    #----------------------------------------------
    ## Sargan test
    
    if num_z == 1:
        if var_ls == 'dum_ls':
            return(res_step1,res_step2,f_test_step1,pvals_ls_endo,[],0)
        else:
            return(res_step1,res_step2,f_test_step1,pvals_ls_endo,[],1)
    else:
        sargan_res = []
        
        for model in res_step2:
            oir = sargan(model.resids, df[righthand_x.split(' + ')], df[righthand_z.split(' + ')])
            
            sargan_res.append(oir.pval)
        
        if var_ls == 'dum_ls':
            return(res_step1,res_step2,f_test_step1,pvals_ls_endo,sargan_res,0)
        else:
            return(res_step1,res_step2,f_test_step1,pvals_ls_endo,sargan_res,1)

#----------------------------------------------
def tableIVtests(num_models,f_test_step1,pvals_ls_endo,sargan_res = None):
    '''Method to create a nice table for the test result of the FDIV.
        The DWH test for endogenous loan sales variable is not included in the
        indicator'''
    
    # Create a list for f-test pvals_ls_endo that are the same length of number of models
    f_test_list = [f_test_step1] * num_models
    pvals_l_endo_list = [j for i in zip(pvals_ls_endo,pvals_ls_endo) for j in i]
    
    if sargan_res:
        indicator_tests = [a*b for a,b in zip([(i > 10) * 1 for i in f_test_list],\
                          [(i > 0.05) * 1 for i in sargan_res])]
        index = ['F-test weak instruments','P-val endogenous w','P-val Sargan','Indicator']
        columns = ['charge_2','allow_2','rwata_2']
        
        return(pd.DataFrame([f_test_list,pvals_l_endo_list,sargan_res,indicator_tests],\
                             index = index, columns = columns))
    else:
        indicator_tests = [a for a in [(i < 0.05) * 1 for i in f_test_list]]
        index = ['T-test weak instruments','P-val endogenous w','Indicator']
        columns = ['charge_2''allow_2','rwata_2']
        
        return(pd.DataFrame([f_test_list,pvals_l_endo_list,indicator_tests],\
                             index = index, columns = columns))        
#----------------------------------------------       
def scoreFDIVtest(test_table):
    '''Makes a test score by summing over all the indicators and dividing that
        number by the number of models'''
    num_cols = test_table.shape[1]
    return(np.sum(test_table.loc['Indicator',:]) / num_cols)
            
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
res_step1 = []
res_step2 = []
f_test_step1 = []
pvals_ls_endo = []
sargan_res = []
list_endo_vars = []

input_gen = itertools.product(vars_endo, list_dfs, vars_z)

for product in input_gen:
    # First set the time dummies (depends on the subset which ones to take)
    time_dummies = ['dum{}'.format(year) for year in product[1].index.get_level_values(1).unique().astype(str).str[:4][1:].tolist()]
                      
    # Run the model
    res_step1_load,res_step2_load,f_test_step1_load,pvals_ls_endo_load,\
    sargan_res_load, endo_var_load =\
    analysesFDIV(*product, righthand_x, time_dummies)
    
    # Save the models
    res_step1.append(res_step1_load)
    res_step2.append(res_step2_load)
    f_test_step1.append(f_test_step1_load)
    pvals_ls_endo.append(pvals_ls_endo_load)
    sargan_res.append(sargan_res_load)
    list_endo_vars.append(endo_var_load)

#----------------------------------------------
# Set things up for nice tables
#----------------------------------------------

# Prelims
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

## Determine regression order
reg_order_step1 = vars_z[0].split(' + ') + righthand_x.split(' + ')
reg_order_step2 = ['G_hat_fd'] + righthand_x.split(' + ')

#----------------------------------------------
'''CHANGE IN CASE OF MULTIPLE OPTIONS FOR Z'''
# Step 1
res_step1_dumls, res_step1_lstotta = split_list(res_step1) 
sum_step1_dumls = summary_col(res_step1_dumls, show = 'se', regressor_order = reg_order_step1)
sum_step1_lstotta = summary_col(res_step1_lstotta, show = 'se', regressor_order = reg_order_step1)

# Step 2
res_step2_dumls, res_step2_lstotta = split_list(res_step2)

res_s2_dumls_shuffled = []
res_s2_lstotta_shuffled = []

for i in range(len(res_step2_dumls[0])):
    for d in range(len(list_dfs)):
        res_s2_dumls_shuffled.append(res_step2_dumls[d][i])
        res_s2_lstotta_shuffled.append(res_step2_lstotta[d][i])

sum_s2_dumls = summary_col(res_s2_dumls_shuffled, show = 'se', regressor_order = reg_order_step2)
sum_s2_lstotta = summary_col(res_s2_lstotta_shuffled, show = 'se', regressor_order = reg_order_step2)

#----------------------------------------------
# Make test vectors to be appended to the tables
## Weak instruments - Step 1
f_test_dumls, f_test_lstotta = split_list(f_test_step1)

## Endo - step 2
endo_dumls, endo_lstotta = split_list(pvals_ls_endo)

### Make the lists to append to table 2
endo_dumls_sort = [vec[i] for i in range(len(res_step2_dumls[0])) for vec in endo_dumls]
endo_lstotta_sort = [vec[i] for i in range(len(res_step2_dumls[0])) for vec in endo_lstotta]

## Sargan test - step 2
sargan_dumls, sargan_lstotta = split_list(sargan_res)

### Make lists to append to table 2
'''NOTE: All uneven items are with hetero correction'''
sargan_dumls_sort = [vec[i] for i in range(len(res_step2_dumls[0])) for vec in sargan_dumls]
sargan_lstotta_sort = [vec[i] for i in range(len(res_step2_dumls[0])) for vec in sargan_lstotta]

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
# Make table for step 1
#----------------------------------------------

## Calculate the adj. R2
k_step1 = [mod.model.exog.shape[0] for mod in res_step1_dumls]

adjrs_dumls = vecAdjR2(list(sum_step1_dumls.tables[2].iloc[2,:].astype(float)),\
                       list(sum_step1_dumls.tables[2].iloc[1,:].astype(float)), k_step1)
adjrs_lstotta = vecAdjR2(list(sum_step1_lstotta.tables[2].iloc[2,:].astype(float)),\
                       list(sum_step1_lstotta.tables[2].iloc[1,:].astype(float)), k_step1)
    
    
## Change the lower part of the table
step1_lower_table_dumls = sum_step1_dumls.tables[2].iloc[[1],:]
step1_lower_table_dumls.loc[-1] = adjrs_dumls
step1_lower_table_dumls.rename({-1:'$Adj. R^2$'}, axis = 'index', inplace = True)
step1_lower_table_dumls.loc[-1] = f_test_dumls
step1_lower_table_dumls.rename({-1:'F-test Weak Instruments'}, axis = 'index', inplace = True)

step1_lower_table_lstotta = sum_step1_lstotta.tables[2].iloc[[1],:]
step1_lower_table_lstotta.loc[-1] = adjrs_lstotta
step1_lower_table_lstotta.rename({-1:'$Adj. R^2$'}, axis = 'index', inplace = True)
step1_lower_table_lstotta.loc[-1] = f_test_lstotta
step1_lower_table_lstotta.rename({-1:'F-test Weak Instruments'}, axis = 'index', inplace = True)

### Add to the table
sum_step1_dumls.tables[2] = step1_lower_table_dumls
sum_step1_lstotta.tables[2] = step1_lower_table_lstotta

## Remove the time dummies
step1_param_table_dumls = sum_step1_dumls.tables[1].iloc[:-np.sum((sum_step1_dumls.tables[1].index.str.contains('dum') * 1))*2,:]
step1_param_table_dumls.loc[-1] = ['Yes'] * len(columns)
step1_param_table_dumls.rename({-1:'Time Dummies'}, axis = 'index', inplace = True)

step1_param_table_lstotta = sum_step1_lstotta.tables[1].iloc[:-np.sum((sum_step1_lstotta.tables[1].index.str.contains('dum') * 1))*2,:]
step1_param_table_lstotta.loc[-1] = ['Yes'] * len(columns)
step1_param_table_lstotta.rename({-1:'Time Dummies'}, axis = 'index', inplace = True)

### add to the table
sum_step1_dumls.tables[1] = step1_param_table_dumls
sum_step1_lstotta.tables[1] = step1_param_table_lstotta

## Make new table
table_step1_dumls = pd.concat(sum_step1_dumls.tables[1:3])
table_step1_lstotta = pd.concat(sum_step1_lstotta.tables[1:3])

### Change the columns of the table
table_step1_dumls.columns = columns
table_step1_lstotta.columns = columns

### Change the index
table_step1_dumls.index = [dict_var_names[key] for key in table_step1_dumls.index]
table_step1_lstotta.index = [dict_var_names[key] for key in table_step1_lstotta.index]

#----------------------------------------------
# Make table for step 2
#----------------------------------------------

## Calculate the adj. R2
k_step2 = [mod.model.exog.shape[0] for mod in res_step1_dumls] * len(dep_vars)

adjrs_step2_dumls = vecAdjR2(list(sum_s2_dumls.tables[2].iloc[2,:].astype(float)),\
                       list(sum_s2_dumls.tables[2].iloc[1,:].astype(float)), k_step2)
adjrs_step2_lstotta = vecAdjR2(list(sum_s2_dumls.tables[2].iloc[2,:].astype(float)),\
                       list(sum_s2_dumls.tables[2].iloc[1,:].astype(float)), k_step2)

## Change the lower part of the table
step2_lower_table_dumls = sum_s2_dumls.tables[2].iloc[[1],:]
step2_lower_table_dumls.loc[-1] = adjrs_step2_dumls
step2_lower_table_dumls.rename({-1:'$Adj. R^2$'}, axis = 'index', inplace = True)
step2_lower_table_dumls.loc[-1] = endo_dumls_sort
step2_lower_table_dumls.rename({-1:'DWH-test'}, axis = 'index', inplace = True)
step2_lower_table_dumls.loc[-1] =  sargan_dumls_sort
step2_lower_table_dumls.rename({-1:'P-val Sargan-test'}, axis = 'index', inplace = True)

step2_lower_table_lstotta = sum_s2_lstotta.tables[2].iloc[[1],:]
step2_lower_table_lstotta.loc[-1] = adjrs_step2_lstotta
step2_lower_table_lstotta.rename({-1:'$Adj. R^2$'}, axis = 'index', inplace = True)
step2_lower_table_lstotta.loc[-1] = endo_lstotta_sort
step2_lower_table_lstotta.rename({-1:'DWH-test'}, axis = 'index', inplace = True)
step2_lower_table_lstotta.loc[-1] =  sargan_lstotta_sort
step2_lower_table_lstotta.rename({-1:'P-val Sargan-test'}, axis = 'index', inplace = True)

### Add to the table
sum_s2_dumls.tables[2] = step2_lower_table_dumls
sum_s2_lstotta.tables[2] = step2_lower_table_lstotta

## Remove the time dummies
step2_param_table_dumls = sum_s2_dumls.tables[1].iloc[:-np.sum((sum_s2_dumls.tables[1].index.str.contains('dum') * 1))*2,:]
step2_param_table_dumls.loc[-1] = ['Yes'] * len(columns) * len(dep_vars)
step2_param_table_dumls.rename({-1:'Time Dummies'}, axis = 'index', inplace = True)

step2_param_table_lstotta = sum_s2_lstotta.tables[1].iloc[:-np.sum((sum_s2_lstotta.tables[1].index.str.contains('dum') * 1))*2,:]
step2_param_table_lstotta.loc[-1] = ['Yes'] * len(columns) * len(dep_vars)
step2_param_table_lstotta.rename({-1:'Time Dummies'}, axis = 'index', inplace = True)

### add to the table
sum_s2_dumls.tables[1] = step2_param_table_dumls
sum_s2_lstotta.tables[1] = step2_param_table_lstotta

## Make new table
table_step2_dumls = pd.concat(sum_s2_dumls.tables[1:3])
table_step2_lstotta = pd.concat(sum_s2_lstotta.tables[1:3])

### Change the columns of the table
columns_step_2 = [('Loan Charge-offs','Full'),('Loan Charge-offs','Pre-Dodd-Frank'),('Loan Charge-offs','Post-Dodd-Frank'),\
                 ('Loan Loss Allowances','Full'),('Loan Loss Allowances','Pre-Dodd-Frank'),('Loan Loss Allowances','Post-Dodd-Frank'),\
                 ('RWA/TA','Full'),('RWA/TA','Pre-Dodd-Frank'),('RWA/TA','Post-Dodd-Frank')]  
table_step2_dumls.columns = pd.MultiIndex.from_tuples(columns_step_2)
table_step2_lstotta.columns = pd.MultiIndex.from_tuples(columns_step_2)

### Change the index
dict_var_names.update({'G_hat_fd':'Dummy Loan Sales'})
table_step2_dumls.index = [dict_var_names[key] for key in table_step2_dumls.index]
dict_var_names.update({'G_hat_fd':'Loan Sales/TA'})
table_step2_lstotta.index = [dict_var_names[key] for key in table_step2_lstotta.index]
    
#----------------------------------------------
# Save the tables 
#----------------------------------------------

# Save the tables
path = r'Results\FD_IV_v5_noendo.xlsx'


with pd.ExcelWriter(path) as writer:
    # Save dumls
    table_step1_dumls.to_excel(writer, sheet_name = 'Step 1 Dumls')

    table_step2_dumls.to_excel(writer, sheet_name = 'Step 2 Dumls')
    
    # Save LS_tot_ta
    table_step1_lstotta.to_excel(writer, sheet_name = 'Step 1 LSTA') 
    
    table_step2_lstotta.to_excel(writer, sheet_name = 'Step 2 LSTA')