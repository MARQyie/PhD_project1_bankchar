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
vars_needed = ['net_coff_on','net_coff_off','net_coff_tot','npl_tot','npl_on', 'npl_off','rwata',\
               'credex_tot','credex_sec','credex_sec_io','credex_sec_subloc', 'credex_nonsec',\
               'reg_lev','reg_cap', 'loanratio', 'roa', 'depratio', 'comloanratio',\
               'mortratio', 'consloanratio','loanhhi',\
               'costinc', 'size', 'bhc','log_empl','log_num_branch','perc_limited_branch']
all_vars = ['IDRSSD','date']

# Remove date > 2017
df = df[df.date < 2017]

# Subset dfs
ls_idrssd = df[df.ls_tot > 0].index.get_level_values(0).unique().tolist() #1646

df_full = df[all_vars + vars_needed]
df_ls = df.loc[df.index.get_level_values(0).isin(ls_idrssd),all_vars + vars_needed]
df_nonls = df.loc[~df.index.get_level_values(0).isin(ls_idrssd),all_vars + vars_needed]

#------------------------------------------------------------
# Make table
#------------------------------------------------------------

# Get summary statistics
ss_full = df_full[vars_needed].describe().T[['mean','std']]
ss_ls = df_ls[vars_needed].describe().T[['mean','std']]
ss_nonls = df_nonls[vars_needed].describe().T[['mean','std']]

# Make comparison table
ss_diff = ss_ls['mean'] - ss_nonls['mean']
ss_perc = ((ss_ls['mean'] - ss_nonls['mean']) / ss_nonls['mean']).round(4).replace(np.inf,'')

# welch's t-test
t_test = []
for var in vars_needed:
    stat, pval = ttest_ind(df_ls[var], df_nonls[var],\
                     equal_var = False, nan_policy = 'omit')
    t_test.append(pval)
ss_pval = pd.Series(t_test, index = vars_needed, name = 'pval')

# Concat tables
ss = pd.concat([ss_full,ss_ls,ss_nonls,ss_diff,ss_perc,ss_pval], axis = 1).round(4)
ss.columns = range(ss.shape[1])

# Add extra stats
## N
stats = [str(df_full.shape[0]), '', str(df_ls.shape[0]), '', str(df_nonls.shape[0]), '', '', '', '']
ss = ss.append(pd.DataFrame(dict(zip(range(ss.shape[1]),stats)), index = ['N']))

## Banks
stats = [str(df_full.IDRSSD.nunique()), '', str(df_ls.IDRSSD.nunique()), '', str(df_nonls.IDRSSD.nunique()), '', '', '', '']
ss = ss.append(pd.DataFrame(dict(zip(range(ss.shape[1]),stats)), index = ['Banks']))

## Years
stats = [str(df_full.date.nunique()), '', str(df_ls.date.nunique()), '', str(df_nonls.date.nunique()), '', '', '', '']
ss = ss.append(pd.DataFrame(dict(zip(range(ss.shape[1]),stats)), index = ['Years']))

# Change name of columns
columns = [('Total Sample', 'Mean'), ('Total Sample', 'SD'),\
        ('Loan Transferrers', 'Mean'), ('Loan Transferrers', 'SD'),\
        ('Non-loan Transferrers', 'Mean'), ('Non-loan Transferrers', 'SD'),\
        ('Difference in Means', 'Abs'),('Difference in Means', '\%'),('Difference in Means', 'P-value')]
ss.columns = pd.MultiIndex.from_tuples(columns)

# Change index
index_col = ['Net Charge-offs (Total)','Net Charge-offs (On)','Net Charge-offs (OBS)',\
             'NPL (Total)','NPL (On)','NPL (OBS)', 'RWATA',\
             'Max. Credit Exp. (Total)','Max. Credit Exp. (Sec.)','Max. Credit Exp. IO (Sec.)',\
             'Max. Credit Exp. Sub./Loc. (Sec.)', 'Max. Credit Exp. (Non-sec.)',\
               'Leverage Ratio','Capital Ratio', 'Loan Ratio', 'ROA', 'Deposit Ratio', 'Commercial Loan Ratio',\
               'Mortgage Ratio', 'Consumer Loan Ratio','Loan HHI',\
               'Cost-to-Income', 'Size', 'BHC','Employees (log)','Number of Branches (log)','Limited Service (\%)',\
               'N','Banks','Years']

ss.index = index_col

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4cm}' + 'p{1cm}' * results.shape[1],
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
    if latex_table.find('N                           &') >= 0:
        size_midrule = '\\midrule '
        location_mid = latex_table.find('N                           &')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
        
    # Add headers for dependent var, ls vars, control vars 
    ## Set strings
    from string import Template
    template_firstsubheader = Template('\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    template_subheaders = Template('& ' * results.shape[1] + '\\\\\n' + '\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    
    txt_distance = template_firstsubheader.substitute(numcols = results.shape[1] + 1, variable = 'Dependent Variables')
    txt_ls = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Recourse Variables')
    txt_loan = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Control Variables')
    txt_lend = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Instrumental Variables')
    
    ## Get locations and insert strings
    location_distance = latex_table.find('Net Charge-offs')
    latex_table = latex_table[:location_distance] + txt_distance + latex_table[location_distance:]
    
    location_ls = latex_table.find('Max. Credit Exp. (Total)')
    latex_table = latex_table[:location_ls] + txt_ls + latex_table[location_ls:]
    
    location_loan = latex_table.find('Leverage Ratio')
    latex_table = latex_table[:location_loan] + txt_loan + latex_table[location_loan:]
    
    location_lend = latex_table.find('Employees')
    latex_table = latex_table[:location_lend] + txt_lend + latex_table[location_lend:]
    
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
size_string = '\\tiny \n'
note = "\\textit{Notes.} Summary statistics of the full sample, loan-transferring and non-loan-transferring banks. Abbreviations sec. and exp. are securitized and exposure, respectively. We compare loan transferrers with non-loan transferrers. Differences in means are calculated with the Welch's t-test for unequal sample sizes and unequal sample variances, only the p-values are given."

ss_latex = resultsToLatex(ss, caption, label,\
                                 size_string = size_string, note_string = note,\
                                 sidewaystable = True)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

ss.to_excel('Tables/Summary_statistics.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_tot_latex.write(ss_latex)
text_ss_tot_latex.close()

#------------------------------------------------------------
# Plots
#------------------------------------------------------------

# Risk variables
## Get the mean variables
df_risk = df[['net_coff_on','npl_on','allow_tot','prov_ratio']].groupby(df.date).mean() * 100

## Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_risk.iloc[:,0], linestyle = '-', color = 'black', label = 'Net Charge-offs')
ax.plot(df_risk.iloc[:,1], linestyle = '--', color = 'black', markersize=14, label = 'NPL')
#ax.plot(df_risk.iloc[:,2], linestyle = ':', color = 'black', label = 'Loan Loss Allowances')
#ax.plot(df_risk.iloc[:,3], linestyle = '-.', color = 'black', markersize=14, label = 'Loan Loss Provisions')
ax.legend()

fig.savefig('Figures\Fig_sumstats_risk_vars_on.png')

# NOTE: We observe similar trends in allow tot and prov_ratio. These are forward-looking measures, which we will use in the robustness checks

# Risk variables off balance sheet
## Get the mean variables
df_risk_bs = df[df.ls_sec > 0][['net_coff_on','net_coff_off','net_coff_tot','npl_on','npl_off','npl_tot']].groupby(df.date).mean() * 100

## Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_risk_bs.iloc[:,1], linestyle = '-', color = 'black', label = 'Net Charge-offs (OBS)')
ax.plot(df_risk_bs.iloc[:,4], linestyle = '--', color = 'black', markersize=14, label = 'NPL (OBS)')
#ax.plot(df_risk.iloc[:,2], linestyle = ':', color = 'black', label = 'Loan Loss Allowances')
#ax.plot(df_risk.iloc[:,3], linestyle = '-.', color = 'black', markersize=14, label = 'Loan Loss Provisions')
ax.legend()

fig.savefig('Figures\Fig_sumstats_risk_vars_off.png')


# Credit exposure variables
## Get the mean variables
df_cred = df[df.credex_tot > 0.0][['credex_tot','credex_sec', 'credex_nonsec']].groupby(df.date).mean() * 100

## Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_cred.iloc[:,0], linestyle = '-', color = 'black', label = 'Credit Exp. (Total)')

fig.savefig('Figures\Fig_sumstats_creditexp.png')
# NOTE: Decline in credit exposure mainly comes from loan sales. Credit exp. sec is much lower, more volatile with only a flat downward trend

# Credit exposure variables securitization only
## Get the mean variables
df_cred_sec = df[df.ls_sec > 0.0][['credex_sec','credex_sec_io','credex_sec_subloc']].groupby(df.date).mean() * 100

## Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_cred_sec.iloc[:,1], linestyle = '-', color = 'black', label = 'Credit Exp. (IO)')
ax.plot(df_cred_sec.iloc[:,2], linestyle = '-.', color = 'black', label = 'Credit Exp. (Sub./Loc.)')
ax.legend()

fig.savefig('Figures\Fig_sumstats_creditexp_sec_only.png')