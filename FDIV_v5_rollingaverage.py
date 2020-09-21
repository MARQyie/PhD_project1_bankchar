#------------------------------------------
# IV treatment model for first working paper
# Rolling average estimation
# Mark van der Plaat
# January 2020

# Import packages
import pandas as pd
import numpy as np

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

# Import method for OLS
from linearmodels import PanelOLS

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

# Import a method to make nice tables
#from Code_docs.help_functions.summary3 import summary_col

# Used for the partial R2s
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale = 2.5, palette = 'Greys_d')

#
from itertools import compress

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
righthand_x = r'RC7204 + loanratio + roa + depratio + comloanratio + mortratio + consloanratio + loanhhi + costinc + RC2170 + bhc'
vars_endo = ['ls_tot_ta'] 
vars_z = ['RIAD4150 + perc_limited_branch'] # In list in case multiple instruments are needed to be run
dep_vars = ['net_coffratio_tot_ta','allowratio_tot_ta','rwata']

# Estimation window
window = 7

#----------------------------------------------
# Load data and add needed variables

# Load df
df = pd.read_csv('Data\df_wp1_clean.csv', index_col = 0)

## Make multi index
df.date = pd.to_datetime(df.date.astype(str))
df.set_index(['IDRSSD','date'],inplace=True)

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

def analysesFDIV(df, var_ls, righthand_x, righthand_z, time_dummies):
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
        return([],[],[])
      
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
    #----------------------------------------------
    
    #----------------------------------------------                                
    ## Weak instruments
    
    if num_z == 1:
        f_test_step1 = res_step1.pvalues[vars_z].values
    else:
        res_step1b = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + time_dummies, data = df).fit(cov_type = 'clustered', cluster_entity = True)
        f_test_step1 = fTestWeakInstruments(df[var_ls], res_step1.fitted_values, res_step1b.fitted_values, num_z)
                
    #----------------------------------------------
    ## Sargan test
    
    if num_z == 1:
        sargan_res = [np.nan] * len(dep_var_step2)
        
    else:
        sargan_res = []
        
        for model in res_step2:
            oir = sargan(model.resids, df[righthand_x.split(' + ')], df[righthand_z.split(' + ')])
            
            sargan_res.append(oir.pval)
    
    # When either test is reject return a one
    if (f_test_step1 < 10):
        tests = [1] * len(dep_var_step2)
    else:
        tests = [(i < 0.05) * 1 for i in sargan_res]

    return(res_step1,res_step2,tests)
           
#----------------------------------------------
#----------------------------------------------
# Analyses
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# Run models
#----------------------------------------------
# Setup the lists that stores the results from analysesFDIV
res_step1 = []
res_step2 = []
res_test = []

# Other prelim
t = df_full_fd.index.get_level_values(1).nunique()
list_yearindex = df.index.get_level_values(1).unique().values

# Loop over the df
for year_vec in range(1,t-window + 2):
    
    ## Subset the data
    subset_boolean = df_full_fd.index.get_level_values(1).isin(list_yearindex[year_vec:year_vec+window])
    data = df_full_fd[subset_boolean]

    ## Set the correct time dummies
    time_dummies = ['dum{}'.format(year) for year in data.index.get_level_values(1).unique().astype(str).str[:4][1:].tolist()]
    
    ## Run the model
    res_step1_load, res_step2_load, res_test_load = \
        analysesFDIV(data, vars_endo[0], righthand_x, vars_z[0],  time_dummies)
    
    ### Save the models
    res_step1.append(res_step1_load)
    res_step2.append(res_step2_load)
    res_test.append(res_test_load)

#----------------------------------------------
# Restructure and prepare the lists for plotting
#----------------------------------------------
# Parameter estimates, standard deviation, test
params = [[], [], []]   
stds =  [[], [], []]   
tests = [[], [], []]   
 
for i in range(3):
    for mod_year, test_year in zip(res_step2, res_test):
        params[i].append(mod_year[i].params)
        stds[i].append(mod_year[i].std_errors)
        tests[i].append(test_year[i])


#----------------------------------------------
# Plot 
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
                 'Time Dummies':'Time Dummies',
                 'G_hat_fd':'Loan Sales/TA'}

# Prelim
year_labels = [list_yearindex[i+int(np.ceil(window/2))] for i in range(t-window+1)]
c = 1.645 #1.960

var_list = r'G_hat_fd + RC7204 + loanratio + roa + depratio + comloanratio + mortratio + consloanratio + loanhhi + costinc + RC2170 + bhc'.split(' + ')
depvar_list = ['LCO','LLA','RWATA']

# Plot
for k in range(len(params)):
    test_estimates = [tests[k][i] for i in range(t-window+1)]
    bar_year = list(compress(year_labels,test_estimates))
    
    for var_name in var_list:
        ## Define lists to plot
        param_estimates = [params[k][i][var_name] for i in range(t-window+1)]
        std_estimates = [stds[k][i][var_name] for i in range(t-window+1)]
        conf_lower = [a - b * c for a,b in zip(param_estimates,std_estimates)]
        conf_upper = [a + b * c for a,b in zip(param_estimates,std_estimates)]
        
        ## Plot prelims 
        fig, ax = plt.subplots(figsize=(12, 8))
        #plt.title(dict_var_names[var_name])
        ax.set(xlabel = 'Mid Year',ylabel = 'Parameter Estimate')
        
        ## Params
        ax.plot(year_labels, param_estimates)
        
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
        fig.savefig('Figures\Moving_averages\{}\MA_{}_{}.png'.format(depvar_list[k],depvar_list[k],var_name))
