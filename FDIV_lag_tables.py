# Make estimation table

''' This script uses the estimation results to make nice estimation tables.
'''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

import numpy as np
import pandas as pd

#------------------------------------------------------------
# Make functions
#------------------------------------------------------------

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
    dictionary = {'Intercept':'Intercept',
                 'G_hat_fd':'Loan Sales',
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
                 'perc_limited':'Limited Service (\%)',
                 'nobs':'Observations',
                 'rsquared':'$R^2$',
                 'f':'F-val',
                 'endo':'DWH-test',
                 'sargan':'P-val Sargan-test'}
    
    # Get parameter column and secondary columns (std, tval, pval)
    params = df.Parameter.iloc[[i for i in range(1,13)] + [0]].round(4)
    
    if show == 'std':
        secondary = df['Std. Err.']
    elif show == 'tval':
        secondary = df['T-stat']
    else:
        secondary = df['P-value']

    # Transform secondary column 
    # If stars, then add stars to the parameter list
    if stars:
        stars_count = ['*' * i for i in sum([df['P-value'] <0.10, df['P-value'] <0.05, df['P-value'] <0.01])]
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
    if step:
        stats = df[['nobs','rsquared','endo','sargan']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    else:
        stats = df[['nobs','rsquared','f']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
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
                               column_format = 'p{2.5cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    return results.to_latex(**function_parameters)


def concatResults(path_list, show = 'pval', stars = False, col_label = None, caption = '', label = '', step = 0):
    '''Calls estimationTable and returns a concatenated table '''
    
    list_of_results = []
    for df_path, lab in zip(path_list, col_label):
        # Read df
        df = pd.read_csv(df_path, index_col = 0)
    
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
    if '|' in results.columns:
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
    note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex
    

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------
    
# Set path list
path_list_step1 = ['Results/Lag/Step_1/Step1_lag_{}.csv'.format(i) for i in range(3)]
path_list_coff_step2 = ['Results/Lag/Step_2/Step2_lag_{}.csv'.format(i) for i in range(3)]
path_list_npl_step2 = ['Results/Lag/Step_2/Step2_lag_{}.csv'.format(i + 3) for i in range(3)]

col_label_step1 = ['({})'.format(i) for i in range(1,len(path_list_step1) + 1)]
col_label_step2 = ['({})'.format(i) for i in range(1,len(path_list_coff_step2) + 1)]

# Set title and label
caption_step1 = 'Estimation Results Benchmark Model (First Stage)'
label_step1 = 'tab:results_benchmark_step1'

caption_step2 = 'Estimation Results Benchmark Model (Second Stage)'
label_step2 = 'tab:results_benchmark_step2'

# Call function
df_results_step1, latex_results_step1 = concatResults(path_list_step1, col_label = col_label_step1,\
                                                  caption = caption_step1, label = label_step1, step = 0)
df_results_coff_step2, latex_results_coff_step2 = concatResults(path_list_coff_step2, col_label = col_label_step2,\
                                                  caption = caption_step2, label = label_step2, step = 1)
df_results_npl_step2, latex_results_npl_step2 = concatResults(path_list_npl_step2, col_label = col_label_step2,\
                                                  caption = caption_step2, label = label_step2, step = 1)

#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results_step1.to_csv('Results/Lag/Step_1/Table_results_lag_step1.csv')

text_file_latex_results = open('Results/Lag/Step_1/Table_results_lag_step1.tex', 'w')
text_file_latex_results.write(latex_results_step1)
text_file_latex_results.close()

df_results_coff_step2.to_csv('Results/Lag/Step_2/Table_results_lag_coff.csv')

text_file_latex_results = open('Results/Lag/Step_2/Table_results_lag_coff.tex', 'w')
text_file_latex_results.write(latex_results_coff_step2)
text_file_latex_results.close()

df_results_npl_step2.to_csv('Results/Lag/Step_2/Table_results_lag_npl.csv')

text_file_latex_results = open('Results/Lag/Step_2/Table_results_lag_npl.tex', 'w')
text_file_latex_results.write(latex_results_npl_step2)
text_file_latex_results.close()
