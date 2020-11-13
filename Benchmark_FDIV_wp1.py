#--------------------------------------------
# Benchmark and FDIV analysis for Working Paper 1
# Mark van der Plaat
# November 2020
#--------------------------------------------

''' This script runs the analysis for working paper 1: cyclicality of recourse.
    The script is an update from previous script and now runs both the benchmark
    and the FDIV model.
    
    The benchmark model is as follows (all in first differences execpt the dummies):
        CR = Beta1 RECOURSE + Beta2 RECOURSE*RECESSION 
             + delta X + alpha + eta_t + epsilon
             
    In the FDIV model, we instrument RECOURSE and RECOURSE*RECESSION 
    
    Dependent variabels:
        1) Net Loan Charge-offs
        2) Non performing loans
    
    For the list of control variables, see paper
    '''

#--------------------------------------------
# Import Packages
#--------------------------------------------
    
# Data manipulation
import pandas as pd
import numpy as np

# Parallelization
import multiprocessing as mp 
from joblib import Parallel, delayed
num_cores = mp.cpu_count()

# Import method for OLS
from linearmodels import PanelOLS

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

#------------------------------------------------------------
# Set up functions 
#------------------------------------------------------------

# Log function
def logVars(data, col):
    return(np.log(data[col] + 1))

# First difference function
def firstDifference(group):
    return(group.diff(periods = 1).dropna())

# F test
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

# Sargan test
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

# Benchmark model function
def benchmarkModel(data, y, x):
   
    # First check the data on column rank
    rank_full = np.linalg.matrix_rank(data[x])
    
    if rank_full != len(x): 
        raise Exception('X is not full column rank')
        
    # Run benchmark model
    model = PanelOLS(data[y], data[x])
    results = model.fit(cov_type = 'clustered', cluster_entity = True)
    
    return results

# Benchmark model function
def FDIV(data, data_z, y, x_endo, x_exo, z):
   
    #--------------------------------------------------------
    rank_full = np.linalg.matrix_rank(data[x_exo + z])
    
    if rank_full != len(x_exo + z): 
        raise Exception('X is not full column rank')
        
    #--------------------------------------------------------
    # First Stage
    ''' We perform two first stages;
        1) Delta RECOURSE on Z and X_exo
        2) Delta RECOURSE * RECESSION on Z and Delta X_exo
        '''
    
    model_11 = PanelOLS(data[x_endo[0]], pd.concat([data_z[z], data[x_exo]], axis = 1))
    model_12 = PanelOLS(data[x_endo[1]], pd.concat([data_z[z], data[x_exo]], axis = 1))
    
    results_11 = model_11.fit(cov_type = 'clustered', cluster_entity = True)
    results_12 = model_12.fit(cov_type = 'clustered', cluster_entity = True)
    
    #--------------------------------------------------------
    # Second Stage
    
    model_2 = PanelOLS(data[y], pd.concat([results_11.fitted_values.rename(columns = {'fitted_values' : 'endo_hat'}),\
                                           results_12.fitted_values.rename(columns = {'fitted_values' : 'endo_int_hat'}),\
                                           data[x_exo]], axis = 1))
    results_2 = model_2.fit(cov_type = 'clustered', cluster_entity = True)
    #--------------------------------------------------------
    # Tests
    '''We test for three things:
        1) The strength of the instrument using a Staiger & Stock type test.\
           F-stat must be > 10.
        2) A DWH test whether dum_ls or ls_tot_ta is endogenous. H0: variable is exogenous
        3) A Sargan test to test the overidentifying restrictions. H0: 
           overidentifying restrictions hold
        
        For only one instrument (num_z == 1) we use the p-val of the instrument in step 1) 
            and we do not do a Sargan test''' 
        
    # Weak instruments (F test)
    ## First run the First stage without instruments and get the residuals
    modelf_11 = PanelOLS(data[x_endo[0]], data[x_exo])
    modelf_12 = PanelOLS(data[x_endo[1]], data[x_exo])
    
    fittedvaluesf_11 = modelf_11.fit(cov_type = 'clustered', cluster_entity = True).fitted_values
    fittedvaluesf_12 = modelf_12.fit(cov_type = 'clustered', cluster_entity = True).fitted_values
    
    # Get the F test
    f_11 = fTestWeakInstruments(data[x_endo[0]], results_11.fitted_values, fittedvaluesf_11, len(z))
    f_12 = fTestWeakInstruments(data[x_endo[1]], results_12.fitted_values, fittedvaluesf_12, len(z))
        
    # DWH test
    model_dwh_1 = PanelOLS(data[y], pd.concat([data[x_endo],\
                                                 results_11.fitted_values.rename(columns = {'fitted_values' : 'endo_hat'}),\
                                           results_12.fitted_values.rename(columns = {'fitted_values' : 'endo_int_hat'}),\
                                           data[x_exo]], axis = 1))
    model_dwh_2 = PanelOLS(data[y], data[x_endo + x_exo])
    
    results_dwh_1 = model_dwh_1.fit(cov_type = 'clustered', cluster_entity = True)
    results_dwh_2 = model_dwh_2.fit(cov_type = 'clustered', cluster_entity = True)
    
    dwh_f = fTestWeakInstruments(pd.concat([results_11.fitted_values.rename(columns = {'fitted_values' : 'endo_hat'}),\
                                           results_12.fitted_values.rename(columns = {'fitted_values' : 'endo_int_hat'})],\
                                           axis = 1), results_dwh_1.fitted_values, results_dwh_2.fitted_values)
        
    dwh_pval = WaldTestStatistic(dwh_f, 'Endogenous regressors are exogenous',\
                                 len(x_endo), results_dwh_1.nobs - results_dwh_1._df_model ,name= 'DWH p-value test')
    

    # Sargan-Hausman test
    if len(z) > len(x_endo):
        sh = sargan(results_2.resids, data[x_exo], data_z[z], len(x_endo))
    else:
        sh = np.nan
    
    return [results_11, results_12, results_2, f_11, f_12, dwh_pval.pval, sh]

# Functions from summary to pandas df
def summaryToDFBenchmark(results):
    # Make a pandas dataframe
    dataframe = pd.read_html(results.summary.tables[1].as_html(), header = 0, index_col = 0)[0]
    
    # Add statistics
    dataframe['nobs'] = results.nobs
    dataframe['rsquared'] = results.rsquared
    
    return dataframe

def summaryToDFFDIV(results, f = None, endo = None, sargan = None, stage = 0):
    # Make a pandas dataframe
    dataframe = pd.read_html(results.summary.tables[1].as_html(), header = 0, index_col = 0)[0]
    
    # Add statistics
    dataframe['nobs'] = results.nobs
    dataframe['rsquared'] = results.rsquared
    
    if not stage:
        dataframe['f'] = f
    elif stage:
        dataframe['endo'] = endo
        dataframe['sargan'] = np.nan
    else:
        pass
    
    return dataframe

# Adj. R2
def adjR2(r2, n, k): 
    ''' Calculates the adj. R2'''
    adj_r2 = 1 - (1 - r2) * ((n - 1)/(n - (k + 1)))
    return(round(adj_r2, 3))

# Latex table methods
def estimationTable(df, show = 'pval', stars = False, col_label = 'Est. Results', step = 0):
    ''' This function takes a df with estimation results and returns 
        a formatted column. 
        
        Input:  df = pandas dataframe estimation results
                show = Str indicating what to show (std, tval, or pval)
                stars = Boolean, show stars based on pval yes/no
                col_label = Str, label for the resulting pandas dataframe
        
        Output: Pandas dataframe
        '''
    # Prelims
    ## Set dictionary for index and columns
    dictionary = {'intercept':'Intercept',
                 'credex_tot':'Recourse',
                 'credex_tot_recession':'Recourse $\times$ Recession',
                 'endo_hat':'Recourse',
                 'endo_int_hat':'Recourse $\times$ Recession',
                 'reg_cap':'Capital Ratio',
                 'loanratio':'Loan Ratio',
                 'roa':'ROA',
                 'depratio':'Deposit Ratio',
                 'comloanratio':'Com. Loan Ratio',
                 'mortratio':'Mortgage Ratio',
                 'consloanratio':'Cons. Loan Ratio',
                 'loanhhi':'Loan HHI',
                 'costinc':'Cost-to-Income',
                 'size':'Size',
                 'bhc':'BHC',
                 'log_empl':'Employees',
                 'log_empl_recession':'Employees $\times$ Recession',
                 'perc_limited_branch':'Limited Service (\%)',
                 'nobs':'Observations',
                 'rsquared':'$R^2$',
                 'adj_rsq':'Adj. $R^2$',
                 'f':'F-val',
                 'endo':'DWH-test',
                 'sargan':'P-val Sargan-test'}
    
    # Get parameter column and secondary columns (std, tval, pval)
    params = df.Parameter.iloc[~df.Parameter.index.str.contains('Intercept|dum')].\
             append(df.Parameter.iloc[df.Parameter.index.str.contains('Intercept')]).round(4)
    
    if show == 'std':
        secondary_val = 'Std. Err.'
    elif show == 'tval':
        secondary_val = 'T-stat'
    else:
        secondary_val = 'P-value'
    secondary = df[secondary_val].iloc[~df[secondary_val].index.str.contains('Intercept|dum')].\
             append(df[secondary_val].iloc[df[secondary_val].index.str.contains('Intercept')]).round(4)

    # Transform secondary column 
    # If stars, then add stars to the parameter list
    if stars:
        pval_list = df['P-value'].iloc[~df['P-value'].index.str.contains('Intercept|dum')].\
             append(df['P-value'].iloc[df['P-value'].index.str.contains('Intercept')]).round(4)
        stars_count = ['*' * i for i in sum([pval_list <0.10, pval_list <0.05, pval_list <0.01])]
        params = ['{:.4f}{}'.format(val, stars) for val, stars in zip(params,stars_count)]
    secondary_formatted = ['({:.4f})'.format(val) for val in secondary]
    
    # Zip lists to make one list
    results = [val for sublist in list(zip(params, secondary_formatted)) for val in sublist]
    
    # Make pandas dataframe
    ## Make index col (make list of lists and zip)
    lol_params = list(zip([dictionary[val] for val in params.index],\
                          ['{} {}'.format(show, val) for val in [dictionary[val] for val in params.index]]))
    index_row = [val for sublist in lol_params for val in sublist]
    
    # Make df
    results_df = pd.DataFrame(results, index = index_row, columns = [col_label])    
    
    # append N, lenders, MSAs, adj. R2, Depvar, and FEs
    ## Make stats lists and maken index labels pretty
    df['adj_rsq'] = adjR2(df.rsquared[0], df.nobs[0], df.Parameter.shape[0])
    
    if step:
        try:
            if df.sargan.isna()[0]:
                stats = df[['nobs','adj_rsq','endo']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
            else:
                stats = df[['nobs','adj_rsq','endo','sargan']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        except:
            stats = df[['nobs','adj_rsq']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    else:
        stats = df[['nobs','adj_rsq','f']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    stats.index = [dictionary[val] for val in stats.index]
    
    ### Make df from stats
    stats_df = pd.DataFrame(stats)
    stats_df.columns = [col_label]
    
    ## Append to results_df
    results_df = results_df.append(stats_df)

    return results_df  

def resultsToLatex(results, caption = '', label = ''):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = False,
                               column_format = 'p{3.5cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    return results.to_latex(**function_parameters)


def concatResults(df_list, show = 'pval', stars = False, col_label = None, caption = '', label = '', step = 0):
    '''Calls estimationTable and returns a concatenated table '''
    
    list_of_results = []
    for df, lab in zip(df_list, col_label):
        # Call estimationTable and append to list
        list_of_results.append(estimationTable(df, show = 'pval', stars = False,\
                                               col_label = lab, step = step))

    # Concat all list of dfs to a single df
    results = pd.concat(list_of_results, axis = 1)
    
    # Order results
    results = results.loc[list_of_results[-1].index.to_numpy(),:]

    # Rename index
    results.index = [result if not show in result else '' for result in results.index]
    
    
    # Rename columns if multicolumn
    if '|' in results.columns[0]:
        col_names = np.array([string.split('|') for string in results.columns])
        results.columns = pd.MultiIndex.from_arrays([col_names[:,0], col_names[:,1]], names = ['Method','Number'])
    
    # To latex
    results_latex = resultsToLatex(results, caption, label)
    
    ## Add table placement
    location = results_latex.find('\begin{table}\n')
    results_latex = results_latex[:location + len('\begin{table}\n') + 1] + '[th!]' + results_latex[location + len('\begin{table}\n') + 1:]
    
    ## Make the font size of the table footnotesize
    size_string = '\\scriptsize \n'
    location = results_latex.find('\centering\n')
    results_latex = results_latex[:location + len('\centering\n')] + size_string + results_latex[location + len('\centering\n'):]
    
    # Add midrule above 'Observations'
    size_midrule = '\\midrule'
    location = results_latex.find('\nObservations')
    results_latex = results_latex[:location] + size_midrule + results_latex[location:]
    
    ## Add note to the table
    # TODO: Add std, tval and stars option
    if step == 1:
        note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level. The dependent variables in column (1) and (3), and (2), and (4) are the net charge-offs ratio, and the non-performing loan ratio, respectively.}\n'
    else:
        note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level. The dependent variables for columns (1) and (2) are Recourse and Recourse $\times$ Recession, respectively.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex

#------------------------------------------------------------
# Load the df and set up
#------------------------------------------------------------

df = pd.read_csv('Data/df_wp1_main.csv')

# Remove date > 2017
df = df[df.date < 2017]

## Make multi index
df.date = pd.to_datetime(df.date.astype(str) + '-12-31')
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------------------------
# Transform data
#------------------------------------------------------------

# Set list with variables to use
vars_y = ['net_coff_tot','npl']
vars_x = ['credex_tot', 'reg_cap', 'loanratio', 'roa', 'depratio',\
          'comloanratio', 'mortratio','consloanratio', 'loanhhi', 'costinc',\
          'size','bhc','intercept']
vars_instr = ['log_empl','log_num_branch','perc_limited_branch', 'log_states']
vars_trans = ['credex_tot_recession'] + ['{}_recession'.format(elem) for elem in vars_instr]
vars_dummies = ['dum' + dummy for dummy in pd.get_dummies(df.index.get_level_values(1)).columns.astype(str).str[:4].tolist()]

# Log the data
if __name__ == '__main__':
    df_log = pd.concat(Parallel(n_jobs = num_cores)(delayed(logVars)(df, col) for col in vars_y + vars_x[:-1] + [vars_instr[2]]), axis = 1)
    
## Add other variables
df_log['log_empl'] = df['log_empl']
df_log['log_num_branch'] = df['log_num_branch']
df_log['log_states'] = df['log_states']

# Take first differences (t - t-1)
df_grouped = df_log.groupby(df.index.get_level_values(0))
    
if __name__ == '__main__':
    df_fd = pd.concat(Parallel(n_jobs = num_cores)(delayed(firstDifference)(group) for name, group in df_grouped))

# Add interaction term  and interacted instruments (based on t)
for var, trans in zip([vars_x[0]] + vars_instr, vars_trans):
    df_fd[trans] = df_fd[var] * (df_fd.index.get_level_values(1).isin([pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)
    df_log[trans] = df_log[var] * (df_log.index.get_level_values(1).isin([pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)

# Add time dummies and intercept
## Add dummies
dummy_fd = pd.DataFrame(np.array(pd.get_dummies(df_fd.index.get_level_values(1))), index = df_fd.index, columns = vars_dummies[1:])
df_fd = pd.concat([df_fd, dummy_fd], axis = 1)

dummy_log = pd.DataFrame(np.array(pd.get_dummies(df_log.index.get_level_values(1))), index = df_log.index, columns = vars_dummies)
df_log = pd.concat([df_log, dummy_log], axis = 1)

## Add intercept
df_fd['intercept'] = 1
df_log['intercept'] = 1

## Drop 2001 in df_log
df_z = df_log[df_log.index.isin(df_fd.index)]

#------------------------------------------------------------
# Benchmark analysis
#------------------------------------------------------------

# Set right-hand-side variables
vars_rhs = [vars_x[0]] + [vars_trans[0]] + vars_x[1:] + vars_dummies[2:]

# Run
if __name__ == '__main__':
    results_benchmark = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_fd, elem, vars_rhs) for elem in vars_y)

#------------------------------------------------------------
# FDIV Analysis
#------------------------------------------------------------
# Set all variables
x_endo = [vars_x[0], vars_trans[0]]
x_exo = vars_x[1:] + vars_dummies[2:]
z = [vars_instr[0], vars_trans[1]]

# Run
if __name__ == '__main__':
    fdiv_stage_11, fdiv_stage_12, fdiv_stage_2, f_11, f_12, dwh, sh = zip(*Parallel(n_jobs = num_cores)(delayed(FDIV)(df_fd, df_z, elem, x_endo, x_exo, z) for elem in vars_y))

#------------------------------------------------------------
# Make neat dataframes and transform to latex
#------------------------------------------------------------

# Make pandas tables of the results
## Benchmark
results_benchmark_list_dfs = []

for result in results_benchmark:
    results_benchmark_list_dfs.append(summaryToDFBenchmark(result))
    
## FDIV
### Step 1
fdiv_stage_1_list_dfs = []

fdiv_stage_1_list_dfs.append(summaryToDFFDIV(fdiv_stage_11[0], f = f_11[0], stage = 0))
fdiv_stage_1_list_dfs.append(summaryToDFFDIV(fdiv_stage_12[0], f = f_12[0], stage = 0))

### Step 2
fdiv_stage_2_list_dfs = []

for result, endo, sar in zip(fdiv_stage_2, dwh, sh):
    fdiv_stage_2_list_dfs.append(summaryToDFFDIV(result, endo = endo, sargan = sar, stage = 1))
    
# To latex
## FDIV Stage 1
### Prelims
col_label_step1 = ['({})'.format(i) for i in range(1,len(fdiv_stage_1_list_dfs) + 1)]
caption_step1 = 'Estimation Results FDIV Model (First Stage)'
label_step1 = 'tab:results_benchmark_step1'

# Get LaTeX results
df_results_step1, latex_results_step1 = concatResults(fdiv_stage_1_list_dfs, col_label = col_label_step1,\
                                                  caption = caption_step1, label = label_step1, step = 0)

# Benchmark and FDIV stage 2
col_label_step2 = ['Benchmark|({})'.format(i) for i in range(1,len(results_benchmark_list_dfs) + 1)] +\
    ['FDIV|({})'.format(i + len(results_benchmark_list_dfs)) for i in range(1,len(fdiv_stage_2_list_dfs) + 1)]
caption_step2 = 'Estimation Results Benchmark Model and FDIV Model (Second Stage)'
label_step2 = 'tab:results_benchmark_fdiv_step2'

df_results_step2, latex_results_step2 = concatResults(results_benchmark_list_dfs + fdiv_stage_2_list_dfs, col_label = col_label_step2,\
                                                  caption = caption_step2, label = label_step2, step = 1)

#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results_step1.to_csv('Results/Main/Step_1/Results_fdiv_step1.csv')

text_file_latex_results = open('Results/Main/Step_1/Table_results_fdiv_step1.tex', 'w')
text_file_latex_results.write(latex_results_step1)
text_file_latex_results.close()

df_results_step2.to_csv('Results/Main/Step_2/Table_results_benchmark_fdiv_step2.csv')

text_file_latex_results = open('Results/Main/Step_2/Table_results_benchmark_fdiv_step2.tex', 'w')
text_file_latex_results.write(latex_results_step2)
text_file_latex_results.close()