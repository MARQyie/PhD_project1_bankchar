#------------------------------------------
# Make figures for first working papers
# Mark van der Plaat
# September 2019 

''' 
    This script creates for working paper #1
    The tables display the mean, median, standard deviation for the total sample
    securitizers and non-securitizers. 

    Data used: US Call reports 2001-2018 year-end
    
    First run: Data_setup_wp1.py and Clean_data_wp1.py
    ''' 
    
#------------------------------------------------------------
# Import packages
#------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale = 1.25, palette = 'Greys_d')

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

from scipy import stats

#------------------------------------------------------------
# Load and set df
#------------------------------------------------------------
df = pd.read_csv('Data\df_wp1_main.csv')

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip())
df.set_index(['IDRSSD','date'],inplace=True)

# Subset the df
sec_idrssd = df[df.ls_sec > 0].index.get_level_values(0).unique().tolist() #305
ls_idrssd = df[df.ls_tot > 0].index.get_level_values(0).unique().tolist() #1574

df_sec = df[df.index.get_level_values(0).isin(sec_idrssd)]
df_nonsec = df[~df.index.get_level_values(0).isin(sec_idrssd)]
df_ls = df[df.index.get_level_values(0).isin(ls_idrssd)]
df_nonls = df[~df.index.get_level_values(0).isin(ls_idrssd)]

#------------------------------------------------------------
# Make Plots
#------------------------------------------------------------
#-----------------------------------------------
# Cyclicality charge-off ratio and NPL ratio
## Make variable

choff_sec = (df_sec.net_coff_tot.mean(level = [1,1]) * 100).droplevel(level = 0)
choff_nonsec = (df_nonsec.net_coff_tot.mean(level = [1,1]) * 100).droplevel(level = 0)
choff_ls = (df_ls.net_coff_tot.mean(level = [1,1]) * 100).droplevel(level = 0)
choff_nonls = (df_nonls.net_coff_tot.mean(level = [1,1]) * 100).droplevel(level = 0)

npl_sec = (df_sec.npl.mean(level = [1,1]) * 100).droplevel(level = 0)
npl_nonsec = (df_nonsec.npl.mean(level = [1,1]) * 100).droplevel(level = 0)
npl_ls = (df_ls.npl.mean(level = [1,1]) * 100).droplevel(level = 0)
npl_nonls = (df_nonls.npl.mean(level = [1,1]) * 100).droplevel(level = 0)


fig, ax = plt.subplots(figsize=(20, 15))
#plt.title('Mean Charge-off Ratio')
ax.set(xlabel='Year', ylabel = 'in %')
#ax.plot(choff_sec_ever, linestyle = '-', color = 'black', label = 'Securitizers')
#ax.plot(choff_nonsec_ever, linestyle = '-.', color = 'black', label = 'Non-Securitizers')
ax.plot(choff_ls, linestyle = '-', color = 'red', label = 'Charge-offs; Loan Sellers')
ax.plot(choff_nonls, linestyle = '-.', color = 'red', label = 'Charge-offs; Non-Loan Sellers')
ax2 = ax.twinx()
#ax2.plot(npl_sec, linestyle = '-', color = 'green', label = 'Loan Ratio; Securitizers')
#ax2.plot(npl_nonsec, linestyle = '-.', color = 'green', label = 'Loan Ratio; Non-Securitizers')
ax2.plot(npl_ls, linestyle = '-', color = 'blue', label = 'NPL Ratio; Loan Sellers')
ax2.plot(npl_nonls, linestyle = '-.', color = 'blue', label = 'NPL Ratio; Non-Loan Sellers')


h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=1)

# Check the changes (first difference)
choff_sec_d = choff_sec.diff().dropna()
choff_nonsec_d = choff_nonsec.diff().dropna()
choff_ls_d = choff_ls.diff().dropna()
choff_nonls_d = choff_nonls.diff().dropna()

npl_sec_d = npl_sec.diff().dropna()
npl_nonsec_d = npl_nonsec.diff().dropna()
npl_ls_d = npl_ls.diff().dropna()
npl_nonls_d = npl_nonls.diff().dropna()


fig, ax = plt.subplots(figsize=(20, 15))
#plt.title('Mean Charge-off Ratio')
ax.set(xlabel='Year', ylabel = 'in %')
#ax.plot(choff_sec_d, linestyle = '-', color = 'black', label = 'Securitizers')
#ax.plot(choff_nonsec_d, linestyle = '-.', color = 'black', label = 'Non-Securitizers')
ax.plot(choff_ls_d, linestyle = '-', marker = 'H', markersize=14, color = 'darkorange', label = 'Charge-offs; Loan Sellers')
ax.plot(choff_nonls_d, linestyle = '-', marker = 'X', markersize=14, color = 'darkorange', label = 'Charge-offs; Non-Loan Sellers')
#ax.plot(npl_sec_d, linestyle = '-', color = 'green', label = 'Loan Ratio; Securitizers')
#ax.plot(npl_nonsec_d, linestyle = '-.', color = 'green', label = 'Loan Ratio; Non-Securitizers')
ax.plot(npl_ls_d, linestyle = ':', marker = 'H', markersize=14, color = 'navy', label = 'NPL Ratio; Loan Sellers')
ax.plot(npl_nonls_d, linestyle = ':', marker = 'X', markersize=14, color = 'navy', label = 'NPL Ratio; Non-Loan Sellers')

ax.legend()

fig.savefig('Figures\Fig_choff_npl_fd_lsvsnonls.png')

# Separate
fig, ax = plt.subplots(figsize=(10, 8))
ax.set(xlabel='Year', ylabel = 'in %')
ax.plot(choff_ls_d, linestyle = '-', marker = 'H', markersize=14, label = 'Loan Sellers')
ax.plot(choff_nonls_d, linestyle = '-', marker = 'X', markersize=14, label = 'Non-Loan Sellers')
ax.legend()
fig.savefig('Figures\Fig_choff_fd_lsvsnonls.png')

fig, ax = plt.subplots(figsize=(10, 8))
ax.set(xlabel='Year', ylabel = 'in %')
ax.plot(npl_ls_d, linestyle = '-', marker = 'H', markersize=14, label = 'Loan Sellers')
ax.plot(npl_nonls_d, linestyle = '-', marker = 'X', markersize=14,label = 'Non-Loan Sellers')
ax.legend()
fig.savefig('Figures\Fig_npl_fd_lsvsnonls.png')