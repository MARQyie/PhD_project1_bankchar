#--------------------------------------------
# Benchmark  analysis for Working Paper 1
# Mark van der Plaat
# November 2020
#--------------------------------------------

''' This script runs the analysis for working paper 1: cyclicality of recourse.
    The script is an update from previous script and now runs the benchmark.
    
    The benchmark model is as follows (all in first differences execpt the dummies):
        CR = Beta1 RECOURSE + Beta2 RECOURSE*RECESSION 
             + delta X + alpha + eta_t + epsilon
    
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

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

#------------------------------------------------------------
# Set up functions 
#------------------------------------------------------------

# Log function
def logVars(data, col):
    return(np.log(data[col] + 1))

# Benchmark model function
def benchmarkModel(data, y, x):
   
    # First check the data on column rank
    rank_full = np.linalg.matrix_rank(data[x])
    
    if rank_full != len(x): 
        raise Exception('X is not full column rank')
        
    # Run benchmark model
    model = PanelOLS(data[y], data[x], entity_effects = True, time_effects = True)
    results = model.fit(cov_type = 'clustered', cluster_entity = True)
    
    return results

# Functions from summary to pandas df
def summaryToDFBenchmark(results):
    # Make a pandas dataframe
    dataframe = pd.read_html(results.summary.tables[1].as_html(), header = 0, index_col = 0)[0]
    
    # Add statistics
    dataframe['nobs'] = results.nobs
    dataframe['rsquared'] = results.rsquared
    
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
                 'credex_nonsec':'Recourse',
                 'credex_nonsec_recession':'Recourse $\times$ Recession',
                 'credex_sec':'Recourse (Sec.)',
                 'credex_sec_recession':'Recourse (Sec.) $\times$ Recession',
                 'credex_tot_dodd':'Recourse $\times$ DFA',
                 'credex_sec_dodd':'Recourse (Sec.) $\times$ DFA',
                 'endo_hat':'Recourse',
                 'endo_int_hat':'Recourse $\times$ Recession',
                 'reg_cap':'Capital Ratio',
                 'eqratio':'Equity Ratio',
                 'loanratio':'Loan Ratio',
                 'loandep':'Loans-to-Deposits',
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
                               multicolumn = False,
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
    results_latex = results_latex[:location + len('\begin{table}\n') + 1] + '[ht]' + results_latex[location + len('\begin{table}\n') + 1:]
    
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
        note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level.}\n'
    else:
        note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level.}\n'
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
#vars_y = ['ls_nonsec','loanlevel','net_coff_on','npl_on','allow_tot','prov_ratio']
vars_y = ['net_coff_on','npl_on','allow_on','prov_ratio','allow_off_rb','allow_off_cea','ddl_off']
#vars_y = ['allow_off_rb','allow_off_items','allow_off_cea','net_coff_on','npl_on','allow_on','prov_ratio']
vars_x = ['credex_sec', 'reg_cap', 'loanratio', 'roa', 'depratio',\
          'comloanratio', 'mortratio','consloanratio', 'loanhhi', 'costinc',\
          'size','bhc']
vars_trans = ['credex_sec_recession', 'credex_sec_dodd']

# Log the data
if __name__ == '__main__':
    df_log = pd.concat(Parallel(n_jobs = num_cores)(delayed(logVars)(df, col) for col in vars_y + vars_x[:-1]), axis = 1)
    
# Add bhc
df_log['bhc'] = df.bhc

# Limit subset to loan sellers
rssdid_lsers = df[df.ls_sec > 0].index.get_level_values(0).unique().tolist()
df_log = df_log[df_log.index.get_level_values(0).isin(rssdid_lsers)]

# Add interaction term  and interacted instruments (based on t)
df_log[vars_trans[0]] = df_log[vars_x[0]] * (df_log.index.get_level_values(1).isin([pd.Timestamp('2001-12-31'), pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)
df_log[vars_trans[1]] = df_log[vars_x[0]] * df[df.index.get_level_values(0).isin(rssdid_lsers)].dodd

# Lag x-vars 
for var in vars_x + vars_trans:
    df_log[var] = df_log.groupby(df_log.index.get_level_values(0))[var].shift(1)

# Take first difference ls_nonsec and loanlevel
#df_log['ls_nonsec'] = df_log.groupby(df_log.index.get_level_values(0))['ls_nonsec'].diff(1)
#df_log['loanlevel'] = df_log.groupby(df_log.index.get_level_values(0))['loanlevel'].diff(1)
    
# Drop na
df_log.dropna(inplace = True)

#------------------------------------------------------------
# Benchmark analysis
#------------------------------------------------------------

# Set right-hand-side variables
vars_rhs_nododd = [vars_x[0]] + [vars_trans[0]]  + vars_x[1:]
vars_rhs_dodd = [vars_x[0]] + [vars_trans[0]] + [vars_trans[1]]  + vars_x[1:]

# Run
if __name__ == '__main__':
    results_nododd = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_nododd) for elem in vars_y)
    results_dodd = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_dodd) for elem in vars_y)

#------------------------------------------------------------
# Make neat dataframes and transform to latex
#------------------------------------------------------------

# Make pandas tables of the results
## Benchmark
results_benchmark_dodd_list_dfs = []
results_benchmark_nododd_list_dfs = []

for result in results_dodd:
    results_benchmark_dodd_list_dfs.append(summaryToDFBenchmark(result))

for result in results_nododd:
    results_benchmark_nododd_list_dfs.append(summaryToDFBenchmark(result))

# Benchmark
col_label_nododd = ['({})'.format(i) for i in range(1,len(results_benchmark_nododd_list_dfs) + 1)]
caption_nododd = 'Estimation Results Robustness Check: Securitization Only'
label_nododd = 'tab:results_robust_nododd'

col_label_dodd = ['({})'.format(i) for i in range(1,len(results_benchmark_dodd_list_dfs) + 1)]
caption_dodd = 'Estimation Results Robustness Check: Dodd-Frank and Securitization'
label_dodd = 'tab:results_robust_dodd'


df_results_dodd, latex_results_dodd = concatResults(results_benchmark_dodd_list_dfs, col_label = col_label_dodd,\
                                                  caption = caption_dodd, label = label_dodd, step = 1)
    

df_results_nododd, latex_results_nododd = concatResults(results_benchmark_nododd_list_dfs, col_label = col_label_nododd,\
                                                  caption = caption_nododd, label = label_nododd, step = 1)


#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results_nododd.to_csv('Robustness_checks/Table_results_robust_nododd.csv')

text_file_latex_results = open('Robustness_checks/Table_results_robust_nododd.tex', 'w')
text_file_latex_results.write(latex_results_nododd)
text_file_latex_results.close()

df_results_dodd.to_csv('Robustness_checks/Table_results_robust_dodd.csv')

text_file_latex_results = open('Robustness_checks/Table_results_robust_dodd.tex', 'w')
text_file_latex_results.write(latex_results_dodd)
text_file_latex_results.close()
