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
    
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Greys', 7)
sns.set_context('notebook')
sns.set(style = 'whitegrid', font_scale = 1.75)

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

# TODO: Change the variable of sec_tot, add non-securitized loan sales and the total group of loan sales

#------------------------------------------
# Split the dataframes on securitizer and non-securitizers
df_sec = df[df.ls_sec_tot > 0]
df_nonsec = df[df.ls_sec_tot == 0]

#------------------------------------------
# Figure 1: Total securitization (B705 + B711; 1-4 Family residential loans + all other)
## Sum per year, drop index level and divide by $1 million
sec_year = df_sec[['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711', 'ls_sec_tot']].sum(level = [1,1])
sec_year = sec_year.droplevel(level = 0)
sec_year = sec_year.divide(1e6)

## Prelims
labels = ['Residential','Home Equity','Credit Card','Auto','Other Consumer','Commercial','All other', 'Total']
line_styles = [(0, (1, 1)),(0, (5, 1)),(0, (3,1,1,1,1,1)),(0, (3,1,1,1)) ,':','-.','--', '-']

##plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitized Loan Sales')
ax.set(xlabel='Year', ylabel = 'Amount of Securitized Loan Sales (in $ Billion)')
for i in range(sec_year.shape[1]):
    ax.plot(sec_year.iloc[:,i], linestyle = line_styles[i], label = labels[i], color = 'black')
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('Fig1a_total_ls_sec.png')

nonsec_year = df[df.ls_nonsec_tot > 0][['RCB790','RCB791','RCB792','RCB793','RCB794','RCB795','RCB796', 'ls_nonsec_tot']].sum(level = [1,1])
nonsec_year = nonsec_year.droplevel(level = 0)
nonsec_year = nonsec_year.divide(1e6)

##plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Non-Securitized Loan Sales')
ax.set(xlabel='Year', ylabel = 'Amount of Non-Securitized Loan Sales (in $ Billion)')
for i in range(nonsec_year.shape[1]):
    ax.plot(nonsec_year.iloc[:,i], linestyle = line_styles[i], label = labels[i], color = 'black')
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('Fig1b_total_ls_nonsec.png')

#-----------------------------------------
# Figure 2: Capital ratios and securitization over TA
'''Plot the securitizers only.'''
## Make variable for Securitization over TA
df_sec['sec_ta'] = df_sec.sec_tot.divide(df.RC2170)

## Prelims
labels = ['Capital/TA','Securitization/TA']
line_styles = ['-','-.']

## Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Capital Ratios and Securitization of Securitizing US Banks')
ax.set(xlabel='Year', ylabel = 'Capital/TA (in %)')
ax.plot(df_sec.eqratio.mean(level = [1,1]).droplevel(level = 0) * 100, linestyle = line_styles[0], label = labels[0], color = 'black')
ax2 = ax.twinx()
ax2.set_ylabel('Securitization/TA (in %)')
ax2.plot(df_sec.sec_ta.mean(level = [1,1]).droplevel(level = 0) * 100, linestyle = line_styles[1], label = labels[1], color = 'black')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=3)
plt.tight_layout()
plt.show()

fig.savefig('Fig2_capta_secta.png')

#-----------------------------------------
# Figure 3: Capital ratios securitizers and non-securitizers
## Prelims
labels = ['Securitizers','Non-Securitizers']
line_styles = ['-','-.']

## Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Capital Ratios of Securitizing and Non-securitizing US Banks')
ax.set(xlabel='Year', ylabel = 'Capital/TA (in %)')
ax.plot(df_sec.eqratio.mean(level = [1,1]).droplevel(level = 0) * 100, linestyle = line_styles[0], label = labels[0], color = 'black')
ax.plot(df_nonsec.eqratio.mean(level = [1,1]).droplevel(level = 0) * 100, linestyle = line_styles[1], label = labels[1], color = 'black')
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('Fig3_capta_sec_nonsec.png')

#-----------------------------------------
# Figure 4: Stacked plot Securitization
## Setup the cumsum array
sec_cumsum = df_sec[['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711']].sum(level = [1,1]).cumsum(axis = 1).droplevel(level = 0).divide(1e6)

## Plot
labels = ['Residential','Home Equity','Credit Card','Auto','Other Consumer','Commercial','All other']

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitization')
ax.set(xlabel='Year', ylabel = 'Amount of securitization (in $ Billion)')
ax.fill_between(sec_cumsum.index, 0, sec_cumsum.iloc[:,0], label = labels[0])
for i in range(sec_cumsum.shape[1]-1):
    ax.fill_between(sec_cumsum.index, sec_cumsum.iloc[:,i], sec_cumsum.iloc[:,i+1], label = labels[i+1])
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Fig4_stacked_cat_sec.png')

#-----------------------------------------
# Figure 5: Concentration of securitizing banks
'''This figure displays the concentration among securitizing banks'''

## Make the array to plot
sec_sum_sort = df_sec.groupby(df_sec.index.get_level_values(0)).sec_tot.sum().sort_values(ascending = False)

groups = [100,75,50,25,10,5]
total_sec = sec_sum_sort.sum()
sec_cons = []

for i in groups:
    sec_cons.append(sec_sum_sort[:i].sum() / total_sec * 100)
