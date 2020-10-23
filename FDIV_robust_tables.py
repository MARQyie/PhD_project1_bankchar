# Make estimation table

''' This script uses the estimation results to make nice estimation tables.
'''
#TODO: FIX TABLES
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

def estimationTable(df, show = 'pval', stars = False, col_label = 'Est. Results', step = 0, check = 0):
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
                 'G_hat_fd':'Recourse (LT/TA)',
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
                 'perc_limited_branch':'Limited Service (\%)',
                 'ls_crisis':'Loan Sales x GFC',
                 'nobs':'Observations',
                 'rsquared':'$R^2$',
                 'f':'F-val',
                 'endo':'DWH-test',
                 'sargan':'P-val Sargan-test'}
    
    if check == 0:
        dictionary['G_hat_fd'] = 'Credit Exposure'
    elif check == 1:
        dictionary['G_hat_fd'] = 'Securitization Recourse'
    elif check == 2:
        dictionary['G_hat_fd'] = 'Loan Sales Recourse'
    
    # Get parameter column and secondary columns (std, tval, pval) 
    # NOTE: We put the intercept last -- not interesting enough to be on top
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
                               column_format = 'p{3.5cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    return results.to_latex(**function_parameters)


def concatResults(path_list, show = 'pval', stars = False, col_label = None, caption = '', label = '', step = 0, check = 0):
    '''Calls estimationTable and returns a concatenated table '''
    
    list_of_results = []
    for df_path, lab, i in zip(path_list, col_label, range(len(path_list))):
        # Read df
        df = pd.read_csv(df_path, index_col = 0)
    
        # Call estimationTable and append to list
        list_of_results.append(estimationTable(df, show = 'pval', stars = False,\
                                               col_label = lab, step = step, check = i))

    # Concat all list of dfs to a single df
    results = pd.concat(list_of_results, axis = 1)
    
    # Order results
    if step == 1:
        ls_vars = list_of_results[0].index[[0,1]].append([res.index[[0,1]] for res in list_of_results if res.index[0] not in list_of_results[0].index]).to_numpy()
        rest_vars = list_of_results[0].index[~list_of_results[0].index.isin(ls_vars)]
    else:
        ls_vars = list_of_results[1].index[[0,1]].append([res.index[[0,1]] for res in list_of_results if res.index[0] not in list_of_results[0].index]).to_numpy()
        rest_vars = list_of_results[1].index[~list_of_results[1].index.isin(ls_vars)]
    
    results = results.loc[np.append(ls_vars,rest_vars),:]

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
    if step == 1:
        note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level.}\n'
    else:
        note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. The model is estimated with clustered standard errors on the bank-level. The F-value in column (1) equals the p-value of the estimated coefficient for employees.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex
    

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------

# Set path list
path_list_step1 = ['Robustness_checks/Step_1/Step1_robust_{}.csv'.format(i) for i in range(0,4)]
path_list_coff_step2 = ['Robustness_checks/Step_2/Step2_robust_{}.csv'.format(i) for i in range(0,4)]
path_list_npl_step2 = ['Robustness_checks/Step_2/Step2_robust_{}.csv'.format(i) for i in range(4,8)]
path_list_rwata_step2 = ['Robustness_checks/Step_2/Step2_robust_{}.csv'.format(i) for i in range(8,12)]    

# Set column labels
col_label_step1 = ['({})'.format(i) for i in range(1,len(path_list_step1) + 1)]
col_label_step2 = ['({})'.format(i) for i in range(1,len(path_list_coff_step2) + 1)]

# Set captions + labels
caption_step1 = 'Estimation Results Robustness Checks (First Stage)'
label_step1 = 'tab:results_robustness_step1'

caption_choff_step2 = 'Estimation Results Robustness Checks Net Charge-offs (Second Stage)'
label_choff_step2 = 'tab:results_robustness_choffs_step2'

caption_npl_step2 = 'Estimation Results Robustness Checks NPL (Second Stage)'
label_npl_step2 = 'tab:results_robustness_npl_step2'

caption_rwata_step2 = 'Estimation Results Robustness Checks RWA/TA (Second Stage)'
label_rwata_step2 = 'tab:results_robustness_rwata_step2'

    # Call function
df_results_step1, latex_results_step1 = concatResults(path_list_step1, col_label = col_label_step1,\
                                                  caption = caption_step1, label = label_step1, step = 0)
df_results_coff_step2, latex_results_coff_step2 = concatResults(path_list_coff_step2, col_label = col_label_step2,\
                                                  caption = caption_choff_step2, label = label_choff_step2, step = 1)
df_results_npl_step2, latex_results_npl_step2 = concatResults(path_list_npl_step2, col_label = col_label_step2,\
                                                  caption = caption_npl_step2, label = label_npl_step2, step = 1)
df_results_rwata_step2, latex_results_rwata_step2 = concatResults(path_list_rwata_step2, col_label = col_label_step2,\
                                                  caption = caption_rwata_step2, label = label_rwata_step2, step = 1)

#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results_step1.to_csv('Robustness_checks/Step_1/Table_results_robust_step1.csv')

text_file_latex_results = open('Robustness_checks/Step_1/Table_results_robust_step1.tex', 'w')
text_file_latex_results.write(latex_results_step1)
text_file_latex_results.close()

df_results_coff_step2.to_csv('Robustness_checks/Step_2/Table_results_robust_coff.csv')

text_file_latex_results = open('Robustness_checks/Step_2/Table_results_robust_coff.tex', 'w')
text_file_latex_results.write(latex_results_coff_step2)
text_file_latex_results.close()

df_results_npl_step2.to_csv('Robustness_checks/Step_2/Table_results_robust_npl.csv')

text_file_latex_results = open('Robustness_checks/Step_2/Table_results_robust_npl.tex', 'w')
text_file_latex_results.write(latex_results_npl_step2)
text_file_latex_results.close()

df_results_rwata_step2.to_csv('Robustness_checks/Step_2/Table_results_robust_rwata.csv')

text_file_latex_results = open('Robustness_checks/Step_2/Table_results_robust_rwata.tex', 'w')
text_file_latex_results.write(latex_results_rwata_step2)
text_file_latex_results.close()


    
'''OLD
for robust in range(4):
    # Set path list
    path_list_step1 = ['Robustness_checks/Step_1/Step1_robust_{}.csv'.format(i + robust * 2) for i in range(2)]
    path_list_coff_step2 = ['Robustness_checks/Step_2/Step2_robust_{}.csv'.format(i + robust * 2) for i in range(2)]
    path_list_npl_step2 = ['Robustness_checks/Step_2/Step2_robust_{}.csv'.format(i  + robust * 2 + 1) for i in range(2)]
    
    col_label_step1 = ['({})'.format(i) for i in range(1,len(path_list_step1) + 1)]
    col_label_step2 = ['({})'.format(i) for i in range(1,len(path_list_coff_step2) + 1)]
    
    # Set title and label
    dic = {0:'Credit Exposure',
           1:'Securitized Loan Sales',
           2:'Non-securitized Loan Sales',
           3:'Loan Sales Dummy',
           4:'Interaction Term'}
    caption_step1 = 'Estimation Results Robustness Check {} (First Stage)'.format(dic[robust])
    label_step1 = 'tab:results_benchmark_step1_{}'.format(robust + 1)
    
    caption_step2 = 'Estimation Results obustness Check {} (Second Stage)'.format(dic[robust])
    label_step2 = 'tab:results_benchmark_step2_{}'.format(robust + 1)
    
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

    df_results_step1.to_csv('Robustness_checks/Step_1/Table_results_robust_step1_{}.csv'.format(robust))
    
    text_file_latex_results = open('Robustness_checks/Step_1/Table_results_robust_step1_{}.tex'.format(robust), 'w')
    text_file_latex_results.write(latex_results_step1)
    text_file_latex_results.close()
    
    df_results_coff_step2.to_csv('Robustness_checks/Step_2/Table_results_robust_coff_{}.csv'.format(robust))
    
    text_file_latex_results = open('Robustness_checks/Step_2/Table_results_robust_coff.tex'.format(robust), 'w')
    text_file_latex_results.write(latex_results_coff_step2)
    text_file_latex_results.close()
    
    df_results_npl_step2.to_csv('Robustness_checks/Step_2/Table_results_robust_npl_{}.csv'.format(robust))
    
    text_file_latex_results = open('Robustness_checks/Step_2/Table_results_robust_npl.tex'.format(robust), 'w')
    text_file_latex_results.write(latex_results_npl_step2)
    text_file_latex_results.close()
'''