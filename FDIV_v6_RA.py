#------------------------------------------
# IV treatment model for first working paper
# Rolling Average estimation
# Mark van der Plaat
# January 2020

# Import packages
import pandas as pd
import numpy as np
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale = 2, palette = 'Greys_d')
from itertools import compress
import datetime

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

# Import method for OLS
from linearmodels import PanelOLS

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

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
        Excel files with the results of the first and second step and the test
        results of a T-test/F-test for weak instruments, a test for endogeneity
        of the Loan Sales variable and a Sargan tests for overidentifying restrictions
        
        A txt-file with a scoring vector containing a score between 0 and 1 
        that scores how good the instrumental variables are working
        
    This script does not create LaTeX tables. 
    '''

#---------------------------------------------- 
#----------------------------------------------
# Prelims
#----------------------------------------------
#----------------------------------------------

# Set the righthand side of the formulas used in analysesFDIV
righthand_x = r'reg_cap + loanratio + roa + depratio + comloanratio + mortratio + consloanratio + loanhhi + costinc + size + bhc'
vars_endo = 'ls_tot' 
vars_z = 'log_empl + perc_limited_branch' # In list in case multiple instruments are needed to be run
dep_vars = ['net_coff_tot','npl']
num_cores = mp.cpu_count()

# Estimation window
window = 7

#----------------------------------------------
# Load data and add needed variables

# Load df
df = pd.read_csv('Data\df_wp1_main.csv')

## Make multi index
df.date = pd.to_datetime(df.date.astype(str) + '-12-31')
df.set_index(['IDRSSD','date'],inplace=True)

#---------------------------------------------------
# Setup the data

## Take log of certain variables
def logVars(col):
    return(np.log(col + 1))
    
log_cols = ['reg_cap', 'loanratio', 'roa', 'depratio', 'comloanratio', 'mortratio',\
            'consloanratio', 'loanhhi', 'costinc', 'ls_tot', 'perc_limited_branch',\
            'net_coff_tot','npl']

for col_name in log_cols:
    df[col_name] = logVars(df[col_name])

## Take the first differences (run parallel)
def firstDIfference(group):
    return(group.diff(periods = 1).dropna())

df_grouped = df[righthand_x.split(' + ') + [vars_endo] + vars_z.split(' + ') + dep_vars].groupby(df.index.get_level_values(0))
    
if __name__ == '__main__':
    df_fd = pd.concat(Parallel(n_jobs = num_cores)(delayed(firstDIfference)(group) for name, group in df_grouped))

## Add time dummies
dummy_fd = pd.get_dummies(df_fd.index.get_level_values(1))

### Add dummies to the dfs
col_dummy = ['dum' + dummy for dummy in dummy_fd.columns.astype(str).str[:4].tolist()]
dummy_fd = pd.DataFrame(np.array(dummy_fd), index = df_fd.index, columns = col_dummy)
df_fd = pd.concat([df_fd, dummy_fd], axis = 1)

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

def analysesFDIV(dep_var, var_ls, df, righthand_z, righthand_x):
    ''' Performs a FDIV linear model. The second stage takes one dependent variable.
        The method also correct for unobserved heterogeneity (see Wooldridge p.). Only 
        allows for one endogenous regressor: var_ls
        
        This method does one df and endogenous regressor.'''
        
    # Prelims
    ## Setup the x and z variable vectors
    vars_x = righthand_x.split(' + ')
    vars_z = righthand_z.split(' + ')
    
    ## Number of z variables
    num_z = righthand_z.count('+') + 1
      
    ## Make timedummies and set a string of the time dummy vector
    time_dummies_list_step1 = ['dum{}'.format(year) for year in df.index.get_level_values(1).unique().astype(str).str[:4][2:].tolist()]
    time_dummies_step1 = ' + '.join(time_dummies_list_step1)
    time_dummies_step2 = time_dummies_step1
    '''
    time_dummies_list_step2 = ['dum{}'.format(year) for year in df.index.get_level_values(1).unique().astype(str).str[:4][1:-1].tolist()]
    time_dummies_step2 = ' + '.join(time_dummies_list_step2)
    '''
  
    #----------------------------------------------    
    # First check the data on column rank
    ## If not full column rank return empty lists
    rank_full = np.linalg.matrix_rank(df[vars_x + vars_z + time_dummies_step1.split(' + ')]) 
    
    if rank_full != len(vars_x + vars_z + time_dummies_step1.split(' + ')):
        return([],[],[],[],[])
      
    #----------------------------------------------
    # STEP 1: First Stage
    #----------------------------------------------
    ''' In the first stage we regress the loan sales variable on the x and z variables
        and the time dummies and calculate the fitted values "G_hat_fd". 
        
        The summary of the first step is returned
        '''  
                          
    # Estimate G_hat
    res_step1 = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + righthand_z + ' + ' + time_dummies_step1   + ' + 1',\
                                      data = df).fit(cov_type = 'clustered', cluster_entity = True)
    
    df['G_hat_fd'] = res_step1.fitted_values
    df['resid_step1'] = res_step1.resids
    
    #----------------------------------------------
    # Step 2: Second Stage
    #----------------------------------------------
    ''' In the second stage we regress all dependent variables respectively on
        the fitted values of step 1 the x variables and the time dummies. In step
        2b we include a correction for unobserved heteroskedasticity.
        '''
    res_step2 = PanelOLS.from_formula(dep_var + ' ~ ' + 'G_hat_fd' + ' + ' + righthand_x +\
                     ' + ' + time_dummies_step2   + ' + 1', data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)    
        
    ''' If we want to lag the independent variables:    
    df_lead = df.copy()
    df_lead[dep_var + '_lead'] = df_lead[dep_var].groupby(df_lead.index.get_level_values(0)).shift(-1)
    df_lead.dropna(inplace = True)
        
    # without correction unobserved heterogeneity
    res_step2 = PanelOLS.from_formula(dep_var + '_lead' + ' ~ ' + righthand_x +\
                     ' + ' + 'G_hat_fd' + ' + ' + time_dummies_step2, data = df_lead).\
                     fit(cov_type = 'clustered', cluster_entity = True)
    '''             
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
        res_step1b = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + time_dummies_step1, data = df).fit(cov_type = 'clustered', cluster_entity = True)
        f_test_step1 = fTestWeakInstruments(df[var_ls], res_step1.fitted_values, res_step1b.fitted_values, num_z)
    
    #----------------------------------------------
    ## Endogenous loan sales variable
    
    res_endo = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                       ' + ' + var_ls + ' + ' + 'resid_step1' + ' + ' + time_dummies_step2, data = df).\
                       fit(cov_type = 'clustered', cluster_entity = True)
    '''
    res_endo = PanelOLS.from_formula(dep_var + '_lead' + ' ~ ' + righthand_x +\
                       ' + ' + var_ls + ' + ' + 'resid_step1' + ' + ' + time_dummies_step2, data = df_lead).\
                       fit(cov_type = 'clustered', cluster_entity = True)
    '''          
    #----------------------------------------------
    ## Sargan test
    
    if num_z == 1:
        return(res_step1,res_step2,f_test_step1,res_endo.pvalues['resid_step1'],[])
        
    else:
        oir = sargan(res_step2.resids, df[righthand_x.split(' + ')], df[righthand_z.split(' + ')])
        '''
        oir = sargan(res_step2.resids, df_lead[righthand_x.split(' + ')], df_lead[righthand_z.split(' + ')])
        '''
        return(res_step1,res_step2,f_test_step1,res_endo.pvalues['resid_step1'],oir.pval)

           
#----------------------------------------------
#----------------------------------------------
# Analyses
#----------------------------------------------
#----------------------------------------------

# Setup parameter values to loop over
t = df_fd.index.get_level_values(1).nunique()
list_yearindex = df.index.get_level_values(1).unique().values

#----------------------------------------------
# Run models
#----------------------------------------------
if __name__ == '__main__': 
    step1, step2, f, endo, sargan_res = zip(*Parallel(n_jobs = num_cores)(delayed(analysesFDIV)(dep_var, vars_endo, df_fd[df_fd.index.get_level_values(1).isin(list_yearindex[year_vec:year_vec+window])], vars_z, righthand_x) for dep_var in dep_vars for year_vec in range(1,t-window + 2)))

#----------------------------------------------
# Save
#----------------------------------------------

# Step 1

def step1ToCSV(table1, fval, i):    
    ## Load the main body of the table
    main_table1 = pd.DataFrame(table1.summary.tables[1])
    main_table1 = main_table1.set_index(main_table1.iloc[:,0])
    main_table1.columns = main_table1.iloc[0,:]
    main_table1 = main_table1.iloc[1:,1:]
    
    ## Add some statistics as columns to the main table
    main_table1['nobs'] = table1.nobs
    main_table1['rsquared'] = table1.rsquared
    main_table1['f'] = fval
    
    ## Save to csv
    main_table1.to_csv('Results\Rolling_average\Step_1\Step1_ra_{}.csv'.format(i))
    
## Run
for table1, fval, i in zip(step1, f, range(len(f))):
    step1ToCSV(table1, fval, i) 
        
# Step 2

def step2ToCSV(table2, endo_val, sargan_val, i):    
    ## Load the main body of the table
    main_table2 = pd.DataFrame(table2.summary.tables[1])
    main_table2 = main_table2.set_index(main_table2.iloc[:,0])
    main_table2.columns = main_table2.iloc[0,:]
    main_table2 = main_table2.iloc[1:,1:]
    
    ## Add some statistics as columns to the main table
    main_table2['nobs'] = table2.nobs
    main_table2['rsquared'] = table2.rsquared
    main_table2['endo'] = endo_val
    main_table2['sargan'] = sargan_val
    
    ## Save to csv
    main_table2.to_csv('Results\Rolling_average\Step_2\Step2_ra_{}.csv'.format(i))

## Run
for table2, endo_val, sargan_val, i in zip(step2, endo, sargan_res, range(len(endo))):
    step2ToCSV(table2, endo_val, sargan_val, i)
    
#----------------------------------------------
#----------------------------------------------
# Make Figures
#----------------------------------------------
#----------------------------------------------
    
# Prelims
year_labels = [list_yearindex[i+int(np.ceil(window/2)) - 1] for i in range(t-window+1)] #subtract one year for the plots
tests_all = [(fval < 10) or (oid < 0.05) for fval, oid in zip(f,sargan_res)]
bar_year_coff = list(compress(year_labels,tests_all[:len(tests_all)//2])) 
bar_year_npl = list(compress(year_labels,tests_all[len(tests_all)//2:])) 
var_names = step2[0]._var_names[:12]

## Make plotting function    
def plotRollingAverage(params, std_errors, depvar, var_name, bar_year):
    global year_labels
    
    ## Prelimns
    c = 1.645
    conf_lower = [a - b * c for a,b in zip(params, std_errors)]
    conf_upper = [a + b * c for a,b in zip(params, std_errors)]
        
    ## Plot prelims 
    fig, ax = plt.subplots(figsize=(12, 8))
    #plt.title(dict_var_names[var_name])
    ax.set(xlabel = 'Mid Year', ylabel = 'Parameter Estimate')
    
    ## Params
    ax.plot(year_labels, params)
    
    ## Stds
    ax.fill_between(year_labels, conf_upper, conf_lower, color = 'deepskyblue', alpha = 0.2)
    
    ## Shaded areas
    ax_limits = ax.get_ylim() # get ax limits
    ax.bar(bar_year,(ax_limits[1] + abs(ax_limits[0])),width = 3.655e2, bottom = ax_limits[0], color = 'dimgrey', alpha = 0.2, linewidth = 0)
    #ax.fill_between(year_labels, ax_limits[1], ax_limits[0], where = test_estimates, step = 'mid', color = 'dimgrey', alpha = 0.2)
    
    ## Accentuate y = 0.0 
    ax.axhline(0, color = 'orangered', alpha = 0.75)
    
    ## Set ax limits
    ax.set_ylim(ax_limits)
    ax.set_xlim([year_labels[0],year_labels[-1]])
    
    ## Last things to do
    plt.tight_layout()

    ## Save the figure
    fig.savefig('Figures\Moving_averages\{}\MA_{}_{}.png'.format(depvar,depvar,var_name))  

# Loop over plotting function
for p in range(12):
    ## Set params and stds
    params_coff = [step2[i].params[p] for i in range(len(step2) // 2)]
    std_errors_coff = [step2[i].std_errors[p] for i in range(len(step2) // 2)]
    params_npl = [step2[i + len(step2) // 2].params[p] for i in range(len(step2) // 2)]
    std_errors_npl = [step2[i + len(step2) // 2].std_errors[p] for i in range(len(step2) // 2)]
    
    # Run models
    plotRollingAverage(params_coff, std_errors_coff, 'net_coff_tot', var_names[p], bar_year_coff)
    plotRollingAverage(params_npl, std_errors_npl, 'npl', var_names[p], bar_year_npl)