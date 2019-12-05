#------------------------------------------
# IV treatment model for first working paper
# Mark van der Plaat
# December 2019 

 # Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

import itertools

# Import method for POLS 
from linearmodels import PanelOLS

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

import sys # to use the help functions needed
sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

# Import a method to make nice tables
from summary3 import summary_col

#--------------------------------------------
# Set parameters 
'''NOTE: log = False performs significantly worse than the log model. No need to 
    toggle off the switch.'''
log = False # If set to False the program estimates the model without logs and with size

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
    df_full = df_full.transform(lambda df: np.log(1 + df))
else:
    df_full[['distance','RIAD4150','num_branch']] = df_full[['distance','RIAD4150','num_branch']].transform(lambda df: np.log(1 + df))

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
            
'''-----------------------------------------''' 
#----------------------------------------------
# Setup the method that does the FDIV plus tests
#----------------------------------------------
'''-----------------------------------------''' 

def analysesFDIV(df, var_ls, righthand_x, righthand_ghat, righthand_z, time_dummies):
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
    '''NOTE: RWATA performed quite bad in previous iterations of the script.
        Hence we remove it from the list'''
    #dep_var_step2 = ['net_coffratio_tot_ta','allowratio_tot_ta','rwata','provratio']
    dep_var_step2 = ['net_coffratio_tot_ta','allowratio_tot_ta','provratio']
    
    ## Make a string of the time dummy vector
    time_dummies = ' + '.join(time_dummies)
    
    ## Determine regression order
    reg_order_step1 = vars_z + vars_x
    reg_order_step2 = ['G_hat_fd'] + vars_x + righthand_ghat.split(' + ')
    
    #----------------------------------------------    
    # First check the data on column rank
    ## If not full column rank return empty lists
    rank_full = np.linalg.matrix_rank(df[vars_x + vars_z + time_dummies.split(' + ')]) 
    
    if rank_full != len(vars_x + vars_z + time_dummies.split(' + ')):
        return([],[],[],[],[],[])
    
    #----------------------------------------------
    # STEP 1: First Stage
    #----------------------------------------------
    ''' In the first stage we regress the loan sales variable on the x and z variables
        and the time dummies and calculate the fitted values "G_hat_fd". We also 
        calculate "G_hat(x-x_bar)" which serves as correction for unobserved hetero-
        skedasticity in step 2b.
        
        The summary of the first step is returned
        '''  
                          
    # Estimate G_hat
    res_step1 = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + righthand_z + ' + ' + time_dummies,\
                                      data = df).fit(cov_type = 'clustered', cluster_entity = True)
    
    df['G_hat_fd'] = res_step1.fitted_values
    
    sum_step1 = summary_col([res_step1], show = 'se', regressor_order = reg_order_step1)

    #----------------------------------------------
    # Calculate G_hat_x_xbar for both first stages
    G_hat_x_xbar_fd = df.loc[:,df.columns.str.contains('_xbar')] * df.G_hat_fd[:, None]

    df[[x + '_G_hat' for x in vars_x]] = G_hat_x_xbar_fd
  
    #----------------------------------------------
    # Step 2: Second Stage
    #----------------------------------------------
    ''' In the second stage we regress all dependent variables respectively on
        the fitted values of step 1 the x variables and the time dummies. In step
        2b we include a correction for unobserved heteroskedasticity.
        
        The summary of all the models in the second step is returned. The list
        "res_step2" is used for the later calculation of tests.
        '''
        
    res_step2 = []
    
    for dep_var in dep_var_step2:
        # without correction unobserved heterogeneity
        res_step2a = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                     ' + ' + 'G_hat_fd' + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
        res_step2.append(res_step2a) # append to results list
             
        # with correction unobserved heterogeneity
        res_step2b = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                     ' + ' + 'G_hat_fd' + '+' + righthand_ghat + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
        res_step2.append(res_step2b) # append to results list
        
    # Create the summary of the models
    sum_step2 = summary_col(res_step2, show = 'se', regressor_order = reg_order_step2)
 
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
                       ' + ' + 'dum_ls' + ' + ' + 'resid_step1' + ' + ' + time_dummies, data = df).\
                       fit(cov_type = 'clustered', cluster_entity = True)
        pvals_ls_endo.append(res_endo.pvalues['resid_step1'])
              
    #----------------------------------------------
    ## Sargan test
    
    if num_z == 1:
        if var_ls == 'dum_ls':
            return(sum_step1,sum_step2,f_test_step1,pvals_ls_endo,[],0)
        else:
            return(sum_step1,sum_step2,f_test_step1,pvals_ls_endo,[],1)
    else:
        sargan_res = []
        
        for model in res_step2:
            oir = sargan(model.resids, df[righthand_x.split(' + ')], df[righthand_z.split(' + ')])
            
            sargan_res.append(oir.pval)
        
        if var_ls == 'dum_ls':
            return(sum_step1,sum_step2,f_test_step1,pvals_ls_endo,sargan_res,0)
        else:
            return(sum_step1,sum_step2,f_test_step1,pvals_ls_endo,sargan_res,1)

#----------------------------------------------
def tableIVtests(num_models,f_test_step1,pvals_ls_endo,sargan_res = None):
    '''Method to create a nice table for the test result of the FDIV.
        The DWH test for endogenous loan sales variable is not included in the
        indicator'''
    
    # Create a list for f-test pvals_ls_endo that are the same length of number of models
    f_test_list = [f_test_step1] * num_models
    pvals_l_endo_list = [j for i in zip(pvals_ls_endo,pvals_ls_endo) for j in i]
    
    if sargan_res:
        '''
        indicator_tests = [a*b*c for a,b,c in zip([(i > 10) * 1 for i in f_test_list],\
                          [(i < 0.05) * 1 for i in pvals_l_endo_list],\
                          [(i > 0.05) * 1 for i in sargan_res])] '''
        indicator_tests = [a*b for a,b in zip([(i > 10) * 1 for i in f_test_list],\
                          [(i > 0.05) * 1 for i in sargan_res])]
        index = ['F-test weak instruments','P-val endogenous w','P-val Sargan','Indicator']
        columns = ['charge_2a','charge_2b','allow_2a','allow_2b','prov_2a','prov_2b',]
        
        return(pd.DataFrame([f_test_list,pvals_l_endo_list,sargan_res,indicator_tests],\
                             index = index, columns = columns))
    else:
        '''
        indicator_tests = [a*b for a,b in zip([(i < 0.05) * 1 for i in f_test_list],\
                          [(i < 0.05) * 1 for i in pvals_l_endo_list])] '''
        indicator_tests = [a for a in [(i < 0.05) * 1 for i in f_test_list]]
        index = ['T-test weak instruments','P-val endogenous w','Indicator']
        columns = ['charge_2a','charge_2b','allow_2a','allow_2b','prov_2a','prov_2b',]
        
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

# Set the righthand side of the formulas used in analysesFDIV
if log:
    righthand_x = r'RC7205 + loanratio + roa + depratio + comloanratio + RC2170 + bhc'
    righthand_ghat = r'RC7205_G_hat + loanratio_G_hat + roa_G_hat + depratio_G_hat + comloanratio_G_hat + RC2170_G_hat + bhc_G_hat'
else:
    righthand_x = r'RC7205 + loanratio + roa + depratio + comloanratio + size + bhc'
    righthand_ghat = r'RC7205_G_hat + loanratio_G_hat + roa_G_hat + depratio_G_hat + comloanratio_G_hat + size_G_hat + bhc_G_hat'

'''NOTE: ls_tot_ta is best performing, so only use this one'''    
#vars_endo = ['dum_ls','ls_tot_ta'] 
vars_endo = ['ls_tot_ta'] 

# Calculate all possible combinations in vars_z
'''OLD
vars_z = ['num_branch','RIAD4150','perc_limited_branch','unique_states','distance']

vars_z_comb = []
for L in range(1, len(vars_z)+1):
    for subset in itertools.combinations(vars_z, L):
        vars_z_comb.append(' + '.join(list(subset)))

z_remove = ['perc_limited_branch','distance','perc_limited_branch + unique_states',\
            'perc_limited_branch + distance','perc_limited_branch + unique_states + distance',\
            'num_branch + perc_limited_branch + unique_states + distance',\
            'unique_states + distance','num_branch + RIAD4150 + unique_states',\
            'num_branch + RIAD4150 + distance','num_branch + perc_limited_branch + unique_states',\
            'num_branch + perc_limited_branch + distance','num_branch + unique_states + distance',\
            'num_branch + RIAD4150 + perc_limited_branch + unique_states',\
            'num_branch + RIAD4150 + perc_limited_branch + distance',\
            'num_branch + RIAD4150 + perc_limited_branch + unique_states + distance',\
            'unique_states','num_branch + RIAD4150', 'num_branch + perc_limited_branch',\
            'num_branch + unique_states','num_branch + distance',\
            'num_branch + RIAD4150 + perc_limited_branch',\
            'num_branch + RIAD4150 + unique_states + distance',\
            'RIAD4150 + perc_limited_branch + unique_states + distance','RIAD4150 + distance',\
            'RIAD4150 + unique_states + distance']

vars_z_comb = [x for x in vars_z_comb if x not in z_remove]   
    '''
vars_z_comb = ['RIAD4150 + perc_limited_branch']
# Setup up th rest of the data for loops
## Note: To loop over the four dataframes, we put them in a list
list_dfs = [df_full_fd, df_pre_fd, df_during_fd,df_post_fd, df_predodd_fd]

#----------------------------------------------
# Run models
#----------------------------------------------
''' Note: To get the results for all four dfs, two loan sale variables and all
    combinations of instrumental variables, we use a triple loop and save the 
    results to a list. 
    '''

# Setup the lists that stores the results from analysesFDIV
res_step1 = []
res_step2 = []
f_test_step1 = []
pvals_ls_endo = []
sargan_res = []
list_endo_vars = []

for data in list_dfs:
    # First set the time dummies (depends on the subset which ones to take)
    time_dummies = ['dum{}'.format(year) for year in data.index.get_level_values(1).unique().astype(str).str[:4][1:].tolist()]
    
    for i in range(len(vars_endo)):
        for z in vars_z_comb:
            res_step1_load,res_step2_load,f_test_step1_load,pvals_ls_endo_load,\
            sargan_res_load, endo_var_load =\
            analysesFDIV(data, vars_endo[i], righthand_x, righthand_ghat, z,  time_dummies)
                
            res_step1.append(res_step1_load)
            res_step2.append(res_step2_load)
            f_test_step1.append(f_test_step1_load)
            pvals_ls_endo.append(pvals_ls_endo_load)
            sargan_res.append(sargan_res_load)
            list_endo_vars.append(endo_var_load)

#----------------------------------------------
# Make test tables and scoring vector
#----------------------------------------------
''' The test tables are later saved to an excel file together with the results
    of step 1 and 2. 
    
    We calculate three different scoring vectors: 1) scoring_vector scores each model
    between 0 and 1, where 1 is the best, 2) scoring_summary summarizes how each
    instrument is doing, 3) scoring_variables scores how well each dependent 
    variable does
    '''
test_tables = []

for f, endo, oir in zip(f_test_step1,pvals_ls_endo,sargan_res):
    test_table_load = tableIVtests(6,f,endo,oir)
    test_tables.append(test_table_load)

scoring_vector = []

for table in test_tables:
    score = scoreFDIVtest(table)
    scoring_vector.append(score)

# Make a summary per instrument
scoring_summary = []
for i in range(len(vars_z_comb)):
    j = list(range(0, len(scoring_vector), len(vars_z_comb)))
    loc_z = [x + i for x in j]
    
    scores_z = [scoring_vector[k] for k in loc_z]
    
    scoring_summary.append(np.sum(scores_z) / len(scores_z))
    
# Make a summary per dependent variable
scoring_variables = []
for table in test_tables:
    scoring_variables.append([np.sum(table.iloc[-1,i*len(vars_endo):i*len(vars_endo)+len(vars_endo)]) for i in range(3)])

'''NOTE: Removed mistake from the calculation: changed the denominator'''
scoring_variables_summary =  np.sum(scoring_variables, axis = 0) / (len(scoring_variables) * len(vars_endo))

# Make a summary per df
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

scoring_dfs = np.sum(list(chunks(scoring_vector, int(len(scoring_vector) / len(list_dfs)))), axis = 1)\
            / (len(scoring_vector) / len(list_dfs))

''' OLD
# Make a summary per loan sales variable
scoring_ls_part = list(chunks(scoring_vector, int(len(scoring_vector) / (len(list_dfs) * len(vars_endo)))))
score_dumls = np.sum(scoring_ls_part[::2]) / (len(scoring_ls_part[::2]) * int(len(scoring_vector) / (len(list_dfs) * len(vars_endo))))
score_lstotta = np.sum(scoring_ls_part[1::2]) / (len(scoring_ls_part[1::2]) * int(len(scoring_vector) / (len(list_dfs) * len(vars_endo))))

scoring_ls = [score_dumls, score_lstotta]
  '''  
#----------------------------------------------
# Save the tables and scoring vector
#----------------------------------------------
# Make a vector containing names for the sheets

names_dfs = ['Full_','Pre_','During_','Post_','PreDodd_']
iter_z = ['{}'.format(i) for i in range(len(vars_z_comb))]
names_dfs_endo = [(a + b) for a in names_dfs for b in vars_endo] 
sheet_names = [(a + b) for a in names_dfs_endo for b in iter_z] 

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
vars_ghat = pd.Series(righthand_ghat.split(' + ')).unique()
vars_x = pd.Series(righthand_x.split(' + ')).unique()

dict_ghat = {}

for key, name in zip(vars_ghat, dict_var_names):
    dict_ghat[key] = '$\hat{{G}}$({})'.format(dict_var_names[name]) 
    
dict_var_names.update(dict_ghat) 

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
    path = r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char\FD_IV_v2_log.xlsx'
else:
    path = r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char\FD_IV_v2.xlsx'

with pd.ExcelWriter(path) as writer:
    for i in range(len(res_step1)):
        
        # Get the index (variable names) from the summary files
        vars_step1 = [x for x in pd.concat(res_step1[i].tables,axis=0).index.tolist() if x not in list_drop]
        vars_step2 = [x for x in pd.concat(res_step2[i].tables,axis=0).index.tolist() if x not in list_drop]
        
        # Add a dict entry for the correct definition of G_hat
        if list_endo_vars[i] == 0:
            dict_var_names.update({'G_hat_fd':'Dummy Loan Sales'})
        else:
            dict_var_names.update({'G_hat_fd':'Loan Sales/TA'})
        
        # Determine the variable names
        var_names_step1 = [dict_var_names[key] for key in vars_step1]
        var_names_step2 = [dict_var_names[key] for key in vars_step2]
        
        # Save the variable names and keys in a dict
        dict_names_step1 = dict(zip(vars_step1,var_names_step1))
        dict_names_step2 = dict(zip(vars_step2,var_names_step2))
        
        # Save the results of step 1 and 2 and the test statistics to the excel file
        res_step1[i].to_excel(writer, sheet_name = sheet_names[i] + '_step1',rename_index = dict_names_step1)
        res_step2[i].to_excel(writer, sheet_name = sheet_names[i] + '_step2', rename_index = dict_names_step2)
        test_tables[i].to_excel(writer, sheet_name = sheet_names[i] + '_tests')

#----------------------------------------------
# Save the scoring vectors
        
## Save the scoring vector
if log:
    path_vec = 'scoring_vector_log.txt'
else:
    path_vec = 'scoring_vector.txt'
    
with open(path_vec, 'w') as filehandle:
    for listitem in scoring_vector:
        filehandle.write('{}\n'.format(listitem))

## Save scoring summary
if log:
    path_sum = 'scoring_summary_log.txt'
else:
    path_sum = 'scoring_summary.txt'
    
with open(path_sum, 'w') as filehandle:
    for name, score in zip(vars_z_comb,scoring_summary):
        filehandle.write('{}: \t {}\n'.format(score, name))
        
## Save variable summary
if log:
    path_varsum = 'variable_summary_log.txt'
else:
    path_varsum = 'variable_summary.txt'
    
with open(path_varsum, 'w') as filehandle:
    for name, score in zip(['net_coffratio_tot_ta','allowratio_tot_ta','provratio'],scoring_variables_summary):
        filehandle.write('{}: \t {}\n'.format(score, name))

## Save scoring dfs
if log:
    path_dfs = 'scoring_dfs_log.txt'
else:
    path_dfs = 'scoring_dfs.txt'
    
with open(path_dfs, 'w') as filehandle:
    for name, score in zip(['Full','Pre-crisis','Crisis','Post-crisis','Pre-Dodd-Frank'],scoring_dfs):
        filehandle.write('{}: \t {}\n'.format(score, name))
'''OLD
## Save scoring ls
if log:
    path_ls = 'scoring_ls_log.txt'
else:
    path_ls = 'scoring_ls.txt'
    
with open(path_ls, 'w') as filehandle:
    for name, score in zip(vars_endo,scoring_ls):
        filehandle.write('{}: \t {}\n'.format(score, name))
'''