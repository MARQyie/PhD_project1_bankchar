# Summary statistics
''' This script returns a table with summary statistics for WP2 '''

import os
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/df_wp1_main.csv')

#------------------------------------------------------------
# Subset dfs
#------------------------------------------------------------

# prelims
vars_needed = ['net_coff_on','npl_on','allow_on','prov_ratio',\
               'credex_tot','reg_lev','reg_cap', 'loanratio', 'roa',\
               'depratio', 'comloanratio','mortratio', 'consloanratio',\
               'loanhhi','costinc', 'size', 'bhc']
all_vars = ['IDRSSD','date']

# Remove date > 2017
df = df[df.date < 2017]

# Subset dfs
ls_idrssd = rssdid_lsers = df[df.ls_tot > 0].IDRSSD.unique().tolist() #1646

df_ls = df.loc[df.IDRSSD.isin(ls_idrssd),all_vars + vars_needed]

# Add crisis dummy
df_ls['recession'] = (df_ls.date.isin([2001,2007,2008,2009]) * 1)

#------------------------------------------------------------
# Make table
#------------------------------------------------------------

# Get summary statistics
ss_ls = df_ls[vars_needed].describe().T[['mean','std']]

# ROund to 4 decimals
ss_ls = ss_ls.round(4)

# Add extra stats
## N
ss_ls.loc['N',:] = [str(df_ls.shape[0]), '']

## Banks
ss_ls.loc['Banks',:] = [str(df_ls.IDRSSD.nunique()), '']

## Years
ss_ls.loc['Years',:] = [str(df_ls.date.nunique()), '']

# Change name of columns
columns = ['Mean', 'SD']
ss_ls.columns = columns

# Change index
index_col = ['Net Charge-offs','NPL','Allowance Ratio','Provision Ratio',\
             'Max. Credit Exp.','Leverage Ratio','Capital Ratio',\
             'Loan Ratio', 'ROA', 'Deposit Ratio', 'Commercial Loan Ratio',\
             'Mortgage Ratio', 'Consumer Loan Ratio','Loan HHI',\
             'Cost-to-Income', 'Size', 'BHC', 'N','Banks','Years']

ss_ls.index = index_col

# Test means different groups
from scipy.stats import ttest_ind

credex_yes = df_ls[df_ls.credex_tot > 0]
credex_no = df_ls[df_ls.credex_tot == 0]

for var in ['net_coff_on','npl_on','allow_on','prov_ratio']:
    t,p = ttest_ind(credex_yes[var], credex_no[var], equal_var = False)
    mean = [credex_yes[var].mean() ,credex_no[var].mean()]
    print(mean, p)


#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    latex_table = results.to_latex(**function_parameters)
       
    # Add string size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + size_string + latex_table[location_size + len('\centering\n'):]
    
    # Add note to the table
    if note_string is not None:
        location_note = latex_table.find('\end{tabular}\n')
        latex_table = latex_table[:location_note + len('\end{tabular}\n')]\
            + '\\begin{tablenotes}\n\\scriptsize\n\\item ' + note_string + '\\end{tablenotes}\n' + latex_table[location_note + len('\end{tabular}\n'):]
            
    # Add midrule above 'Observations'
    if latex_table.find('N                     &') >= 0:
        size_midrule = '\\midrule '
        location_mid = latex_table.find('N                     &')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
        
    # Add headers for dependent var, ls vars, control vars 
    ## Set strings
    from string import Template
    template_firstsubheader = Template('\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    template_subheaders = Template('& ' * results.shape[1] + '\\\\\n' + '\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    
    txt_distance = template_firstsubheader.substitute(numcols = results.shape[1] + 1, variable = 'Dependent Variables')
    txt_ls = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Recourse Variables')
    txt_loan = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Control Variables')
    
    ## Get locations and insert strings
    location_distance = latex_table.find('Net Charge-offs')
    latex_table = latex_table[:location_distance] + txt_distance + latex_table[location_distance:]
    
    location_ls = latex_table.find('Max. Credit Exp.')
    latex_table = latex_table[:location_ls] + txt_ls + latex_table[location_ls:]
    
    location_loan = latex_table.find('Leverage Ratio')
    latex_table = latex_table[:location_loan] + txt_loan + latex_table[location_loan:]
    
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('{table}','{sidewaystable}',2)
        
    # Make threeparttable
    location_centering = latex_table.find('\centering\n')
    latex_table = latex_table[:location_centering + len('\centering\n')] + '\\begin{threeparttable}\n' + latex_table[location_centering + len('\centering\n'):]
    
    location_endtable = latex_table.find('\\end{tablenotes}\n')
    latex_table = latex_table[:location_endtable + len('\\end{tablenotes}\n')] + '\\end{threeparttable}\n' + latex_table[location_endtable + len('\\end{tablenotes}\n'):]
        
    return latex_table

# Call function
caption = 'Summary Statistics'
label = 'tab:summary_statistics'
size_string = '\\scriptsize \n'
note = "\\textit{Notes.} Summary statistics of the full sample. The abbreviation exp. stands for exposure."

ss_latex = resultsToLatex(ss_ls, caption, label,\
                                 size_string = size_string, note_string = note,\
                                 sidewaystable = False)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

ss_ls.to_excel('Tables/Summary_statistics.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_tot_latex.write(ss_latex)
text_ss_tot_latex.close()

#------------------------------------------------------------
# Plots
#------------------------------------------------------------

# Risk variables
## Get the mean variables
df_risk = df_ls[['net_coff_on','npl_on','allow_on','prov_ratio']].groupby(df.date).mean() * 100

## Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_risk.iloc[:,0], linestyle = '-', color = 'black', label = 'Net Charge-offs')
ax.plot(df_risk.iloc[:,1], linestyle = '--', color = 'black', markersize=14, label = 'NPL')
ax.legend()
plt.tight_layout()

fig.savefig('Figures\Fig_sumstats_risk_vars_realized.png')

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_risk.iloc[:,2], linestyle = ':', color = 'black', label = 'Loan Loss Allowances')
ax.plot(df_risk.iloc[:,3], linestyle = '-.', color = 'black', markersize=14, label = 'Loan Loss Provisions')
ax.legend()
plt.tight_layout()

fig.savefig('Figures\Fig_sumstats_risk_vars_anticipated.png')

# NOTE: We observe similar trends in allow tot and prov_ratio. These are forward-looking measures, which we will use in the robustness checks


# Credit exposure variables
## Get the mean variables
df_cred = df_ls[['credex_nonsec']].groupby(df.date).mean() * 100

## Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_cred.iloc[:,0], linestyle = '-', color = 'black', label = 'Credit Exp. (Total)')
plt.tight_layout()

fig.savefig('Figures\Fig_sumstats_creditexp.png')