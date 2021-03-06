#--------------------------------------------
# Robustness Checks for Working Paper 1
# Mark van der Plaat
# November 2020
#--------------------------------------------

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

# Iteration tools
from itertools import islice

# Plot packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale = 2, palette = 'Greys_d')

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
                 'credex_tot_alt':'Recourse',
                 'credex_tot_alt_recession':'Recourse $\times$ Recession',
                 'credex_sec':'Recourse (Sec.)',
                 'credex_sec_recession':'Recourse (Sec.) $\times$ Recession',
                 'credex_nonsec':'Recourse (Loan Sales)',
                 'credex_nonsec_recession':'Recourse (Loan Sales) $\times$ Recession',
                 'credex_sbo':'Recourse (SBO)',
                 'credex_sbo_recession':'Recourse (SBO) $\times$ Recession',
                 'credex_tot_vix_mean':'Recourse $\times$ VIX',
                 'credex_tot_pc_mean':'Recourse $\times$ PC',
                 'credex_tot_gdp':'Recourse $\times$ $\Delta$GDP',
                 'credex_tot_dodd':'Recourse $\times$ DFA',
                 'credex_sec_dodd':'Recourse (Sec.) $\times$ DFA',
                 'credex_nonsec_dodd':'Recourse (Loan Sales) $\times$ DFA',
                 'credex_tot_alt_dodd':'Recourse $\times$ DFA',
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
                               column_format = 'p{4cm}' + 'p{1cm}' * results.shape[1],
                               escape = False,
                               multicolumn = False,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    return results.to_latex(**function_parameters)


def concatResults(df_list, show = 'pval', stars = False, col_label = None, caption = '', label = '', step = 0, sidewaystable = False):
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
    size_string = '\\tiny \n'
    location = results_latex.find('\centering\n')
    results_latex = results_latex[:location + len('\centering\n')] + size_string + results_latex[location + len('\centering\n'):]
    
    # Add midrule above 'Observations'
    size_midrule = '\\midrule'
    location = results_latex.find('\nObservations')
    results_latex = results_latex[:location] + size_midrule + results_latex[location:]
    
    ## Add note to the table
    # TODO: Add std, tval and stars option
    if step == 1:
        note_string = '\\begin{tablenotes}\n\\scriptsize\n\item\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level.\\end{tablenotes}\n'
    else:
        note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    # Makes sidewaystable
    if sidewaystable:
        results_latex = results_latex.replace('{table}','{sidewaystable}',2)
        
    # Make threeparttable
    location_centering = results_latex.find('\centering\n')
    results_latex = results_latex[:location_centering + len('\centering\n')] + '\\begin{threeparttable}\n' + results_latex[location_centering + len('\centering\n'):]
    
    location_endtable = results_latex.find('\\end{tablenotes}\n')
    results_latex = results_latex[:location_endtable + len('\\end{tablenotes}\n')] + '\\end{threeparttable}\n' + results_latex[location_endtable + len('\\end{tablenotes}\n'):]
    
    
    return results,results_latex

# Rolling window 
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# List slicer
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# Plotting function rolling averages
def plotRollingAverage(params, std_errors, depvar, var_name):
    
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
   
    ## Accentuate y = 0.0 
    ax.axhline(0, color = 'orangered', alpha = 0.75)
    
    ## Set ax limits
    ax_limits = ax.get_ylim()
    ax.set_ylim(ax_limits)
    ax.set_xlim([year_labels[0],year_labels[-1]])
    
    ## Last things to do
    plt.tight_layout()

    ## Save the figure
    fig.savefig('Figures\Moving_averages\{}\MA_{}_{}.png'.format(depvar,depvar,var_name))  
    
#------------------------------------------------------------
# Load the df and set up
#------------------------------------------------------------

# Main dataset
df = pd.read_csv('Data/df_wp1_main.csv')

## Remove date > 2017
df = df[df.date < 2017]

## Make multi index
df.date = pd.to_datetime(df.date.astype(str) + '-12-31')
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------------------------
# Transform data
#------------------------------------------------------------

# Set list with variables to use
vars_y = ['net_coff_on','allow_on','prov_ratio','allow_off_rb','allow_off_cea','ddl_off']

## y-vars for npl
vars_y_npl = ['npl_on','npl90_on','nplna_on','npl_res_on','restruc_loans',\
              'npl_nores_on','npl_re_on','npl_ci_on','npl_he_on','npl_oth_on']

## X vars for rolling window, alternative business cycle
vars_x = ['credex_tot', 'reg_cap', 'loanratio', 'roa', 'depratio',\
          'comloanratio', 'mortratio','consloanratio', 'loanhhi', 'costinc',\
          'size','bhc']

## X vars alternative Recourse
vars_x_recalt1 = ['credex_tot_alt', 'reg_cap', 'loanratio', 'roa', 'depratio',\
          'comloanratio', 'mortratio','consloanratio', 'loanhhi', 'costinc',\
          'size','bhc']
vars_x_recalt2 = ['credex_nonsec','credex_sec','credex_sbo', 'reg_cap', 'loanratio', 'roa', 'depratio',\
          'comloanratio', 'mortratio','consloanratio', 'loanhhi', 'costinc',\
          'size','bhc']

# All interaction terms
vars_trans = ['credex_tot_recession', 'credex_tot_vix_mean',\
              'credex_tot_tfp','credex_tot_pc_mean',\
              'credex_tot_cs_mean','credex_tot_gdp','credex_tot_rig_mean',\
              'credex_tot_alt_recession','credex_nonsec_recession',\
              'credex_sec_recession','credex_sbo_recession','credex_tot_dodd']

# Log the data
if __name__ == '__main__':
    df_log = pd.concat(Parallel(n_jobs = num_cores)(delayed(logVars)(df, col) for col in vars_y  + vars_y_npl + np.unique(vars_x[:-1] + vars_x_recalt1[:-1] + vars_x_recalt2[:-1]).tolist()), axis = 1)
    
# Add bhc
df_log['bhc'] = df.bhc

# Limit subset to loan sellers
rssdid_lsers = df[df.ls_tot > 0].index.get_level_values(0).unique().tolist()
df_log = df_log[df_log.index.get_level_values(0).isin(rssdid_lsers)]

# Make credex_tot_alt dummy
df_log['credex_tot_alt'] = (df_log.credex_tot_alt > 0) * 1

# Add interaction term  and interacted instruments (based on t)
## With Recession
df_log[vars_trans[0]] = df_log[vars_x[0]] * (df_log.index.get_level_values(1).isin([pd.Timestamp('2001-12-31'), pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)
df_log[vars_trans[7]] = df_log[vars_x_recalt1[0]] * (df_log.index.get_level_values(1).isin([pd.Timestamp('2001-12-31'), pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)
df_log[vars_trans[8]] = df_log[vars_x_recalt2[0]] * (df_log.index.get_level_values(1).isin([pd.Timestamp('2001-12-31'), pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)
df_log[vars_trans[9]] = df_log[vars_x_recalt2[1]] * (df_log.index.get_level_values(1).isin([pd.Timestamp('2001-12-31'), pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)
df_log[vars_trans[10]] = df_log[vars_x_recalt2[2]] * (df_log.index.get_level_values(1).isin([pd.Timestamp('2001-12-31'), pd.Timestamp('2007-12-31'), pd.Timestamp('2008-12-31'), pd.Timestamp('2009-12-31')]) * 1)

## Dodd Frank
df_log[vars_trans[11]] = df_log[vars_x[0]] * df[df.index.get_level_values(0).isin(rssdid_lsers)].dodd

## Other BS measures
df_log[vars_trans[1]] = df_log[vars_x[0]] * np.log(df[df.index.get_level_values(0).isin(rssdid_lsers)].vix_mean)   
df_log[vars_trans[2]] = df_log[vars_x[0]] * df[df.index.get_level_values(0).isin(rssdid_lsers)].tfp
df_log[vars_trans[3]] = df_log[vars_x[0]] * df[df.index.get_level_values(0).isin(rssdid_lsers)].pc_mean
df_log[vars_trans[4]] = df_log[vars_x[0]] * df[df.index.get_level_values(0).isin(rssdid_lsers)].cs_mean
df_log[vars_trans[5]] = df_log[vars_x[0]] * df[df.index.get_level_values(0).isin(rssdid_lsers)].gdp
df_log[vars_trans[6]] = df_log[vars_x[0]] * df[df.index.get_level_values(0).isin(rssdid_lsers)].rig_mean

# Lag x-vars 
for var in np.unique(vars_x[:-1] + vars_x_recalt1[:-1] + vars_x_recalt2[:-1]).tolist() + vars_trans:
    df_log[var] = df_log.groupby(df_log.index.get_level_values(0))[var].shift(1)
    
# Drop na
df_log.dropna(inplace = True)

#------------------------------------------------------------
# Robust
#------------------------------------------------------------

# Set right-hand-side variables
## Rolling window
vars_rhs_rw = vars_x

## Alternative Business Cycle measures
vars_rhs_vix_mean = [vars_x[0]] + [vars_trans[1]] + vars_x[1:]
vars_rhs_pc = [vars_x[0]] + [vars_trans[3]] + vars_x[1:]
vars_rhs_gdp = [vars_x[0]] + [vars_trans[5]] + vars_x[1:]

## Alternative Recourse measures
vars_rhs_altrec1 = [vars_x_recalt1[0]] + [vars_trans[7]] +  vars_x_recalt1[1:]
vars_rhs_altrec2 = [vars_x_recalt2[0]] + [vars_trans[8]] + [vars_x_recalt2[1]] + [vars_trans[9]] + [vars_x_recalt2[2]] + [vars_trans[10]] +  vars_x_recalt2[3:]

## Dodd-Frank
vars_rhs_dfa = [vars_x[0]] + [vars_trans[0]] + [vars_trans[11]] + vars_x[1:]

## NPL
vars_rhs_npl = [vars_x[0]] + [vars_trans[0]] + vars_x[1:]

# Get dattes for Rolling window
dates = df_log.index.get_level_values('date').year.unique()

# Run
if __name__ == '__main__':
    ## Rolling window
    results_rw = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log[df_log.index.get_level_values('date').year.isin(roll_window)], elem, vars_rhs_rw) for elem in vars_y for roll_window in window(dates, 5))
    
    ## Alternative Business Cycle measures
    results_vix = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_vix_mean) for elem in vars_y)
    results_pc = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_pc) for elem in vars_y)
    results_gdp = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_gdp) for elem in vars_y)
    
    ## Alternative Recourse measures
    results_altrec1 = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_altrec1) for elem in vars_y)
    results_altrec2 = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_altrec2) for elem in vars_y)
    
    ## Dodd-Frank Act
    results_dfa = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_dfa) for elem in vars_y)
    
    ## NPL
    results_npl = Parallel(n_jobs = num_cores)(delayed(benchmarkModel)(df_log, elem, vars_rhs_npl) for elem in vars_y_npl)

#------------------------------------------------------------
# Make neat dataframes and transform to latex
#------------------------------------------------------------

# Make pandas tables of the results
## Alternative Business Cycle measures
results_vix_list_dfs = []
results_pc_list_dfs = []
results_gdp_list_dfs = []

for result_vix, result_pc, result_gdp in zip(results_vix, results_pc, results_gdp):
    results_vix_list_dfs.append(summaryToDFBenchmark(result_vix))
    results_pc_list_dfs.append(summaryToDFBenchmark(result_pc))
    results_gdp_list_dfs.append(summaryToDFBenchmark(result_gdp))

## Alternative Recourse measures
results_altrec1_list_dfs = []
results_altrec2_list_dfs = []

for result_altrec1, result_altrec2 in zip(results_altrec1, results_altrec2):
    results_altrec1_list_dfs.append(summaryToDFBenchmark(result_altrec1))
    results_altrec2_list_dfs.append(summaryToDFBenchmark(result_altrec2))

## Dodd-Frank Act
results_dfa_list_dfs = []

for result_dfa in  results_dfa:
    results_dfa_list_dfs.append(summaryToDFBenchmark(result_dfa))
    
## NPL
results_npl_list_dfs = []

for result_npl in  results_npl:
    results_npl_list_dfs.append(summaryToDFBenchmark(result_npl))


# Make dfs and latex tables
col_label = ['({})'.format(i) for i in range(1,len(results_vix_list_dfs) + 1)]
col_label_npl = ['({})'.format(i) for i in range(1,len(results_npl_list_dfs) + 1)]

## Alternative Business Cycle measures
caption_vix = 'Robustness Check Alternative Business Cycle Measures: VIX'
label_vix = 'tab:results_robust_vix'

caption_pc = 'Robustness Check Alternative Business Cycle Measures: Producer Confidence'
label_pc = 'tab:results_robust_pc'

caption_gdp = 'Robustness Check Alternative Business Cycle Measures: GDP Growth'
label_gdp = 'tab:results_robust_gdp'

## Alternative Recourse measures
caption_altrec1 = 'Robustness Check Alternative Recourse Measures: Recourse Dummy'
label_altrec1 = 'tab:results_robust_altrec1'

caption_altrec2 = 'Robustness Check Alternative Recourse Measures: Recourse Split'
label_altrec2 = 'tab:results_robust_altrec2'

## Dodd-Frank Act
caption_dfa= 'Robustness Check: The Dodd-Frank Act'
label_dfa = 'tab:results_robust_dfa'

## NPL
caption_npl = 'Robustness Check: Non-performing Loans'
label_npl = 'tab:results_robust_npl'

## Make
df_results_vix, latex_results_vix = concatResults(results_vix_list_dfs, col_label = col_label,\
                                                  caption = caption_vix, label = label_vix, step = 1)
df_results_pc, latex_results_pc = concatResults(results_pc_list_dfs, col_label = col_label,\
                                                  caption = caption_pc, label = label_pc, step = 1)
df_results_gdp, latex_results_gdp = concatResults(results_gdp_list_dfs, col_label = col_label,\
                                                  caption = caption_gdp, label = label_gdp, step = 1)
    
df_results_altrec1, latex_results_altrec1 = concatResults(results_altrec1_list_dfs, col_label = col_label,\
                                                  caption = caption_altrec1, label = label_altrec1, step = 1)
df_results_altrec2, latex_results_altrec2 = concatResults(results_altrec2_list_dfs, col_label = col_label,\
                                                  caption = caption_altrec2, label = label_altrec2, step = 1)
    
df_results_dfa, latex_results_dfa = concatResults(results_dfa_list_dfs, col_label = col_label,\
                                                  caption = caption_dfa, label = label_dfa, step = 1)
    
df_results_npl, latex_results_npl = concatResults(results_npl_list_dfs, col_label = col_label_npl,\
                                                  caption = caption_npl, label = label_npl, step = 1,\
                                                  sidewaystable = True)

#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

## Alternative Business Cycle measures
df_results_vix.to_csv('Robustness_checks/Table_results_robust_vix.csv')
text_file_latex_results = open('Robustness_checks/Table_results_robust_vix.tex', 'w')
text_file_latex_results.write(latex_results_vix)
text_file_latex_results.close()

df_results_pc.to_csv('Robustness_checks/Table_results_robust_pc.csv')
text_file_latex_results = open('Robustness_checks/Table_results_robust_pc.tex', 'w')
text_file_latex_results.write(latex_results_pc)
text_file_latex_results.close()

df_results_gdp.to_csv('Robustness_checks/Table_results_robust_gdp.csv')
text_file_latex_results = open('Robustness_checks/Table_results_robust_gdp.tex', 'w')
text_file_latex_results.write(latex_results_gdp)
text_file_latex_results.close()

## Alternative Recourse measures
df_results_altrec1.to_csv('Robustness_checks/Table_results_robust_altrec1.csv')
text_file_latex_results = open('Robustness_checks/Table_results_robust_altrec1.tex', 'w')
text_file_latex_results.write(latex_results_altrec1)
text_file_latex_results.close()

df_results_altrec2.to_csv('Robustness_checks/Table_results_robust_altrec2.csv')
text_file_latex_results = open('Robustness_checks/Table_results_robust_altrec2.tex', 'w')
text_file_latex_results.write(latex_results_altrec2)
text_file_latex_results.close()

##Dodd-Frank Act
df_results_dfa.to_csv('Robustness_checks/Table_results_robust_dfa.csv')
text_file_latex_results = open('Robustness_checks/Table_results_robust_dfa.tex', 'w')
text_file_latex_results.write(latex_results_dfa)
text_file_latex_results.close()

## NPL
df_results_npl.to_csv('Robustness_checks/Table_results_robust_npl.csv')
text_file_latex_results = open('Robustness_checks/Table_results_robust_npl.tex', 'w')
text_file_latex_results.write(latex_results_npl)
text_file_latex_results.close()

#------------------------------------------------------------
# Plot the Rolling averages
#------------------------------------------------------------

# Set labels and test vect
year_labels = [i[len(i)//2] for i in window(dates, 5)]
var_names = results_rw[0]._var_names

# Get chunks
coff, allow, prov, allow_obs_rb, allow_obs_cea, ddl = chunkIt(results_rw,6)

for p in range(len(var_names)):
    ## Get params and stds
    params_coff = [mod.params[p] for mod in coff]
    params_allow = [mod.params[p] for mod in allow]
    params_prov = [mod.params[p] for mod in prov]
    params_allow_obs_cea = [mod.params[p] for mod in allow_obs_cea]
    params_allow_obs_rb = [mod.params[p] for mod in allow_obs_rb]
    params_ddl = [mod.params[p] for mod in ddl]
    
    std_errors_coff = [mod.std_errors[p] for mod in coff]
    std_errors_allow = [mod.std_errors[p] for mod in allow]
    std_errors_prov = [mod.std_errors[p] for mod in prov]
    std_errors_allow_obs_cea = [mod.std_errors[p] for mod in allow_obs_cea]
    std_errors_allow_obs_rb = [mod.std_errors[p] for mod in allow_obs_rb]
    std_errors_ddl = [mod.std_errors[p] for mod in ddl]
    
    # Run models
    plotRollingAverage(params_coff, std_errors_coff, 'net_coff', var_names[p])
    plotRollingAverage(params_allow, std_errors_allow, 'allow', var_names[p])
    plotRollingAverage(params_prov, std_errors_prov, 'prov', var_names[p])
    plotRollingAverage(params_allow_obs_cea, std_errors_allow_obs_cea, 'allow_obs_cea', var_names[p])
    plotRollingAverage(params_allow_obs_rb, std_errors_allow_obs_rb, 'allow_obs_rb', var_names[p])
    plotRollingAverage(params_ddl, std_errors_ddl, 'ddl', var_names[p])
