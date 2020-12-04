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
sns.set(style = 'white', font_scale = 1.75)

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/df_wp1_main.csv')

#------------------------------------------------------------
# Subset dfs
#------------------------------------------------------------

# prelims
vars_needed = ['net_coff_on','npl_on','allow_on','prov_ratio',\
               'allow_off_rb','allow_off_cea','ddl_off',\
               'credex_tot','reg_cap', 'loanratio', 'roa',\
               'depratio', 'comloanratio','mortratio', 'consloanratio',\
               'loanhhi','costinc', 'size', 'bhc',\
               'vix_mean','pc_mean','gdp']
all_vars = ['IDRSSD','date']

# Remove date > 2017
df = df[df.date < 2017]

# Subset dfs
ls_idrssd = rssdid_lsers = df[df.ls_tot > 0].IDRSSD.unique().tolist() #1646

df_ls = df.loc[df.IDRSSD.isin(ls_idrssd),all_vars + vars_needed]

# Add crisis dummy
df_ls['recession'] = (df_ls.date.isin([2001,2007,2008,2009]) * 1)

#------------------------------------------------------------
# Full Summary statistics
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
             'OBS Allowance Ratio (Asset Eq.) ','OBS Allowance Ratio (Credit Eq.)',\
             'OBS Loan Delinq. and Defaults',\
             'Max. Credit Exp.','Capital Ratio',\
             'Loan Ratio', 'ROA', 'Deposit Ratio', 'Commercial Loan Ratio',\
             'Mortgage Ratio', 'Consumer Loan Ratio','Loan HHI',\
             'Cost-to-Income', 'Size', 'BHC','VIX','Prod. Confidence','$\Delta$GDP', 'N','Banks','Years']

ss_ls.index = index_col

#------------------------------------------------------------
# Make table that shows the difference between Recourse and 
# no recourse banks
#------------------------------------------------------------

# Get means different groups
credex_yes = df_ls[df_ls.credex_tot > 0]
credex_no = df_ls[df_ls.credex_tot == 0]

risk_vars = ['net_coff_on','npl_on','allow_on','prov_ratio',\
               'allow_off_rb','allow_off_cea','ddl_off']

ss_credex = pd.DataFrame([credex_no[risk_vars].mean(),credex_yes[risk_vars].mean()]).T

# Add index and columns
ss_credex.index = ['Net Charge-offs','NPL','Allowance Ratio','Provision Ratio',\
             'OBS Allowance Ratio (Asset Eq.) ','OBS Allowance Ratio (Credit Eq.)',\
             'OBS Loan Delinq. and Defaults']
ss_credex.columns = ['No Recourse','Recourse']

# Add difference column
ss_credex.loc[:,'$\Delta$'] = ss_credex.iloc[:,0] - ss_credex.iloc[:,1]

# Test means different groups
p_list = []
for var in risk_vars:
    t,p = ttest_ind(credex_no[var], credex_yes[var], equal_var = False)
    p_list.append(p)
    
# Add boldface when difference is p<0.05
ss_credex = ss_credex.round(4).astype(str)
ss_credex.iloc[:,2] = ['\textbf{' + j + '}' if i < .05 else j for i,j in zip(p_list,ss_credex.iloc[:,2])]

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False, full = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{5cm}' + 'p{2cm}' * results.shape[1],
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
    if full:
        if latex_table.find('N                                &') >= 0:
            size_midrule = '\\midrule '
            location_mid = latex_table.find('N                                &')
            latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
            
        # Add headers for dependent var, ls vars, control vars 
        ## Set strings
        from string import Template
        template_firstsubheader = Template('\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
        template_subheaders = Template('& ' * results.shape[1] + '\\\\\n' + '\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
        
        txt_distance = template_firstsubheader.substitute(numcols = results.shape[1] + 1, variable = 'Dependent Variables')
        txt_ls = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Recourse Variables')
        txt_loan = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Control Variables')
        txt_bs = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Business Cycle Variables')
        
        ## Get locations and insert strings
        location_distance = latex_table.find('Net Charge-offs')
        latex_table = latex_table[:location_distance] + txt_distance + latex_table[location_distance:]
        
        location_ls = latex_table.find('Max. Credit Exp.')
        latex_table = latex_table[:location_ls] + txt_ls + latex_table[location_ls:]
        
        location_loan = latex_table.find('Capital Ratio')
        latex_table = latex_table[:location_loan] + txt_loan + latex_table[location_loan:]
        
        location_bs = latex_table.find('VIX')
        latex_table = latex_table[:location_bs] + txt_bs + latex_table[location_bs:]
    
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
                                 sidewaystable = False, full = True)
    
caption_credex = 'Difference Banks With and Without Recourse'
label_credex = 'tab:sumstat_diffrecourse'
note_credex = "\\textit{Notes.} We compare the means of banks with and without recourse for all BS and OBS risk measures. Difference means $p<0.05$ in boldface. To test the difference in means we use a t-test for unequal sample size and variance The abbreviation Eq. stands for equivalent, and Delinq. stands for Delinquencies."

ss_credex_latex = resultsToLatex(ss_credex, caption_credex, label_credex,\
                                 size_string = size_string, note_string = note_credex,\
                                 sidewaystable = False, full = False)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

ss_ls.to_excel('Tables/Summary_statistics.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_tot_latex.write(ss_latex)
text_ss_tot_latex.close()

ss_credex.to_excel('Tables/Difference_banks_recourse.xlsx')

text_ss_tot_latex = open('Tables/Difference_banks_recourse.tex', 'w')
text_ss_tot_latex.write(ss_credex_latex)
text_ss_tot_latex.close()


#------------------------------------------------------------
# Plots
#------------------------------------------------------------

# Risk variables
## Get the mean variables
df_risk = df_ls[['net_coff_on','npl_on','allow_on','prov_ratio','allow_off_rb','allow_off_cea','ddl_off']].groupby(df.date).mean() * 100

## Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_risk.iloc[:,0], linestyle = '-', color = 'black', label = 'Net Charge-offs')
ax.plot(df_risk.iloc[:,1], linestyle = '--', color = 'black', markersize=14, label = 'NPL')
ax.legend(loc=4)
plt.tight_layout()

fig.savefig('Figures\Fig_sumstats_risk_vars_realized.png')

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_risk.iloc[:,2], linestyle = ':', color = 'black', label = 'Loan Loss Allowances')
ax.plot(df_risk.iloc[:,3], linestyle = '-.', color = 'black', markersize=14, label = 'Loan Loss Provisions')
ax.legend(loc=4)
plt.tight_layout()

fig.savefig('Figures\Fig_sumstats_risk_vars_anticipated.png')

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
lns1 = ax.plot(df_risk.iloc[:,4], linestyle = ':', color = 'black', label = 'OBS Allow. Ratio (Asset Eq.; left axis)')
ax2 = ax.twinx()
lns2 = ax2.plot(df_risk.iloc[:,5], linestyle = '-', color = 'black', markersize=14, label = 'OBS Allow. Ratio (Credit Eq.; right axis)')
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=4)
plt.tight_layout()

fig.savefig('Figures\Fig_sumstats_risk_vars_obs_allow.png')

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Average (in %)')
ax.plot(df_risk.iloc[:,6], linestyle = '--', color = 'black', markersize=14, label = 'OBS Loan Delinq. and Defaults (left axis)')
ax.legend(loc=4)
plt.tight_layout()

fig.savefig('Figures\Fig_sumstats_risk_vars_obs_ddl.png')


# NOTE: We observe similar trends in allow tot and prov_ratio. These are forward-looking measures, which we will use in the robustness checks