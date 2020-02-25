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
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

import os
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

from scipy import stats

#------------------------------------------
# Load df
df = pd.read_csv('Data\df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip())
df.set_index(['IDRSSD','date'],inplace=True)

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
#plt.title('Total Securitized Loan Sales')
ax.set(xlabel='Year', ylabel = 'Amount of Securitized Loan Sales (in $ Billion)')
for i in range(sec_year.shape[1]):
    ax.plot(sec_year.iloc[:,i], linestyle = line_styles[i], label = labels[i], color = 'black')
ax.axvline(pd.Timestamp('2009-12-31'), color = 'r', alpha = 0.75)
plt.text(pd.Timestamp('2009-12-31'), 160, ' Adoption Dodd-Frank Act', fontsize = 15, alpha = 0.75)
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('Figures\Fig1a_total_ls_sec.png')

nonsec_year = df[df.ls_nonsec_tot > 0][['RCB790','RCB791','RCB792','RCB793','RCB794','RCB795','RCB796', 'ls_nonsec_tot']].sum(level = [1,1])
nonsec_year = nonsec_year.droplevel(level = 0)
nonsec_year = nonsec_year.divide(1e6)

##plot
fig, ax = plt.subplots(figsize=(12, 8))
#plt.title('Total Non-Securitized Loan Sales')
ax.set(xlabel='Year', ylabel = 'Amount of Non-Securitized Loan Sales (in $ Billion)')
for i in range(nonsec_year.shape[1]):
    ax.plot(nonsec_year.iloc[:,i], linestyle = line_styles[i], label = labels[i], color = 'black')
ax.axvline(pd.Timestamp('2009-12-31'), color = 'r', alpha = 0.75)
plt.text(pd.Timestamp('2009-12-31'), 22, ' Adoption Dodd-Frank Act', fontsize = 15, alpha = 0.75)
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('Figures\Fig1b_total_ls_nonsec.png')

#-----------------------------------------
# Figure 2: Capital ratios and securitization over TA
'''Plot the securitizers only.'''
## Make variable for Securitization over TA
df_sec['sec_ta'] = df_sec.ls_sec_tot.divide(df.RC2170)

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

fig.savefig('Figures\Fig2_capta_secta.png')

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

fig.savefig('Figures\Fig3_capta_sec_nonsec.png')

#-----------------------------------------
# Figure 4: Stacked plot Securitizated and nonsecuritized loan sales
## Setup the cumsum array
sec_cumsum = df_sec[['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711']].sum(level = [1,1]).cumsum(axis = 1).droplevel(level = 0).divide(1e6)
nonsec_cumsum = df_sec[['RCB790','RCB791','RCB792','RCB793','RCB794','RCB795','RCB796']].sum(level = [1,1]).cumsum(axis = 1).droplevel(level = 0).divide(1e6)


hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']

## Sec Plot
labels = ['Residential','Home Equity','Credit Card','Auto','Other Consumer','Commercial','All other']

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitizated Loan Sales')
ax.set(xlabel='Year', ylabel = 'Amount of securitization (in $ Billion)')
ax.fill_between(sec_cumsum.index, 0, sec_cumsum.iloc[:,0], label = labels[0], hatch = hatch[0])
for i in range(sec_cumsum.shape[1]-1):
    ax.fill_between(sec_cumsum.index, sec_cumsum.iloc[:,i], sec_cumsum.iloc[:,i+1], label = labels[i+1], hatch = hatch[i+1])
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig4a_stacked_cat_secls.png')

## Nonsec Plot
labels = ['Residential','Home Equity','Credit Card','Auto','Other Consumer','Commercial','All other']

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitizated Loan Sales')
ax.set(xlabel='Year', ylabel = 'Amount of securitization (in $ Billion)')
ax.fill_between(nonsec_cumsum.index, 0, nonsec_cumsum.iloc[:,0], label = labels[0], hatch = hatch[0])
for i in range(nonsec_cumsum.shape[1]-1):
    ax.fill_between(nonsec_cumsum.index, nonsec_cumsum.iloc[:,i], nonsec_cumsum.iloc[:,i+1], label = labels[i+1], hatch = hatch[i+1])
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig4b_stacked_cat_nonsecls.png')

#-----------------------------------------
# Figure 5: Plot the allowance ratios

allow_sum = df[df.RCONB557 > 0.0][['allowratio_on_on','allowratio_off_on','allowratio_tot_on']].mean(level = [1,1]).droplevel(level = 0).multiply(1e2)

## Plot
labels = ['On-Balance','Off-Balance','On + Off-balance']
line_styles = ['-','-.',':']

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Allowance Ratio for On-, Off-Balance and Total Toans')
ax.set(xlabel='Year', ylabel = 'On-balance and Total Loan Sales (in %)')
ax.plot(allow_sum.iloc[:,0], linestyle = line_styles[0], label = labels[0], color = 'black')
ax.plot(allow_sum.iloc[:,2], linestyle = line_styles[2], label = labels[2], color = 'black')
ax.grid(False)

ax2 = ax.twinx()
ax2.set(xlabel='Year', ylabel = 'Off-balance Loan Sales (in %)')
ax2.plot(allow_sum.iloc[:,1], linestyle = line_styles[1], label = labels[1], color = 'black')
ax2.grid(False)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=3)
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig5_allowance_rates.png')

#-----------------------------------------
# Figure 6: Credit exposure loan sales
# Make the variables
credex_secls = df[df.ls_sec_tot > 0]['lsseccredex_ratio'].mean(level = [1,1]).droplevel(level = 0).multiply(1e2)
credex_nonsecls = df[df.ls_nonsec_tot > 0]['lsnonseccredex_ratio'].mean(level = [1,1]).droplevel(level = 0).multiply(1e2)
credex_ls = df[df.ls_tot > 0]['lscredex_ratio'].mean(level = [1,1]).droplevel(level = 0).multiply(1e2)

credex_sum = [credex_secls,credex_nonsecls,credex_ls]

## Plot
labels = ['Securitized','Non-Securitized','Total']
line_styles = ['-','-.',':']

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Maximum Credit Exposure of Loan Sales to Loan Sales')
ax.set(xlabel='Year', ylabel = 'Maximum Credit Exposure (in %)')
for i in range(3):
    ax.plot(credex_sum[i], linestyle = line_styles[i], label = labels[i], color = 'black')
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig6a_credex_ratio.png')

#-----------------------------------------
## Different scaling
credex_secls_alt = (df[df.ls_sec_tot > 0].ls_sec_credex / df[df.ls_sec_tot > 0].RC2170).replace(np.inf, 0).mean(level = [1,1]).droplevel(level = 0).multiply(1e2)
credex_nonsecls_alt = (df[df.ls_nonsec_tot > 0].ls_nonsec_credex / df[df.ls_nonsec_tot > 0].RC2170).replace(np.inf, 0).mean(level = [1,1]).droplevel(level = 0).multiply(1e2)
credex_ls_alt = (df[df.ls_tot > 0].ls_credex / df[df.ls_tot > 0].RC2170).replace(np.inf, 0).mean(level = [1,1]).droplevel(level = 0).multiply(1e2)
allowratio_alt = (df.RIAD3123 / df.RC2170).replace(np.inf, 0).mean(level = [1,1]).droplevel(level = 0).multiply(1e2)

credex_sum_alt = [credex_secls_alt,credex_nonsecls_alt,credex_ls_alt,allowratio_alt]

labels = ['Sec. Loan Sales','Non-Sec. Loan Sales','Total Loan Sales', 'Allowance']
line_styles = ['-','-.',':']

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Maximum Credit Exposure of Loan Sales to Total Assets')
ax.set(xlabel='Year', ylabel = 'Maximum Credit Exposure to TA (in %)')
for i in range(3):
    ax.plot(credex_sum_alt[i], linestyle = line_styles[i], label = labels[i], color = 'black')
ax.grid(False)
    
ax2 = ax.twinx()
ax2.set(xlabel='Year', ylabel = 'Allowance to TA (in %)')
ax2.plot(credex_sum_alt[3], linestyle = line_styles[i], label = labels[i], color = 'black')
ax2.grid(False)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=3)
#ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig6b_credex_ratio_alt.png')
#-----------------------------------------
# Figure 7: Concentration of securitizing banks
'''This figure displays the concentration among loan selling banks in 2018'''
'''
## Make the array to plot
ls_sum_sort = df[(df.ls_tot > 0) & (df.index.get_level_values(1) == '2018-12-30')].ls_tot.sort_values(ascending = False)

groups = [100,75,50,25,10,5]
total_ls = ls_sum_sort.sum()
ls_cons = []

for i in groups:
    ls_cons.append(ls_sum_sort[:i].sum() / total_ls * 100)
    
x_values = ['100 banks','75 banks','50 banks','25 banks','10 banks','5 banks']

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Amount of Loan Sales: $61.3 billion') 
ax.set(xlabel = 'Percentage of Total Loan Sales (in %)')
ax.barh(x_values, ls_cons, color = 'grey')
for i in range(len(groups)):
    ax.text(ls_cons[i] + 0.5, i, str(round(ls_cons[i],1)), fontweight = 'bold', verticalalignment='center')
fig.tight_layout()

fig.savefig('Figures\Fig7_concentration_banks.png')

#-----------------------------------------
# NOT INFORMATIVE
# Figure : Top 10 banks in 2018
## Select the top 10 banks
top_10_sec = df[df.index.get_level_values(1) == '2018-12-30'].ls_sec_tot.divide(df.RC2170).multiply(1e2).sort_values(ascending = False).iloc[:10]
top_10_nonsec = df[df.index.get_level_values(1) == '2018-12-30'].ls_nonsec_tot.divide(df.RC2170).multiply(1e2).sort_values(ascending = False).iloc[:10]

## Add the IDRSSDs
name_list = df.name

top_10_sec = pd.concat([top_10_sec, name_list], axis = 1, join = 'inner').sort_values(by = 0, ascending = False)
top_10_nonsec = pd.concat([top_10_nonsec, name_list], axis = 1, join = 'inner').sort_values(by = 0, ascending = False)

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Top 10 Banks Loan Sales-to-Total Assets') 
ax.set(xlabel = 'Percentage of Total Assets (in %)')
ax.bar(top_10_sec.name, top_10_sec.iloc[:,0])
plt.xticks(rotation='vertical')
'''
#-----------------------------------------
# Figure 8: Number of Branches vs. LS/TA

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Number of Branches vs. Loan Sales') 
ax.set(ylabel = 'Number of Branches',xlabel = 'Loan Sales-to-Total Assets (in %)')
ax.scatter(df.ls_sec_tot.divide(df.RC2170).multiply(1e2)[df.ls_sec_tot > 0.0], df[df.ls_sec_tot > 0.0].num_branch,\
           color = 'black', marker = 'v', label = 'Securitized Loan Sales',\
           alpha = 0.5)
ax.scatter(df.ls_nonsec_tot.divide(df.RC2170).multiply(1e2)[df.ls_nonsec_tot > 0.0], \
           df[df.ls_nonsec_tot > 0.0].num_branch, color = 'grey', marker = '8', label = 'Non-securitized Loan Sales',
           alpha = 0.5)
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig8_numbranch_lsta.png')

#-----------------------------------------
# Figure 9: Cap/TA vs. LS/TA
## For the regression lines
slope_sec, intercept_sec,_ ,_ ,_ = stats.linregress(df.ls_sec_tot.divide(df.RC2170).multiply(1e2)[df.ls_sec_tot > 0.0], \
                                   df[df.ls_sec_tot > 0.0].eqratio)
slope_nonsec, intercept_nonsec,_ ,_ ,_ = stats.linregress(df.ls_nonsec_tot.divide(df.RC2170).multiply(1e2)[df.ls_nonsec_tot > 0.0], \
                                   df[df.ls_nonsec_tot > 0.0].eqratio)

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Capital-to-Total Assets vs. Loan Sales (in %)') 
ax.set(ylabel = 'Capital-to-Total Assets',xlabel = 'Loan Sales-to-Total Assets (in %)')
ax.scatter(df.ls_sec_tot.divide(df.RC2170).multiply(1e2)[df.ls_sec_tot > 0.0], df[df.ls_sec_tot > 0.0].eqratio,\
           color = 'black', marker = 'v', label = 'Securitized Loan Sales',\
           alpha = 0.5)
ax.plot(df.ls_sec_tot.divide(df.RC2170).multiply(1e2)[df.ls_sec_tot > 0.0],\
        slope_sec*df.ls_sec_tot.divide(df.RC2170).multiply(1e2)[df.ls_sec_tot > 0.0]+intercept_sec,\
        color = 'black', label = 'Securitized Loan Sales')

ax.scatter(df.ls_nonsec_tot.divide(df.RC2170).multiply(1e2)[df.ls_nonsec_tot > 0.0], \
           df[df.ls_nonsec_tot > 0.0].eqratio, color = 'grey', marker = '8', label = 'Non-securitized Loan Sales',
           alpha = 0.5)
ax.plot(df.ls_nonsec_tot.divide(df.RC2170).multiply(1e2)[df.ls_nonsec_tot > 0.0],\
        slope_nonsec*df.ls_nonsec_tot.divide(df.RC2170).multiply(1e2)[df.ls_nonsec_tot > 0.0]+intercept_nonsec,\
        color = 'grey', linestyle = '-.', label = 'Non-securitized Loan Sales')
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig9_capta_lsta.png')

#-----------------------------------------
# Figure 10: Credit derivatives/TA vs. LS/TA
## Make the dfs
ls_year = df['ls_tot'].sum(level = [1,1])
ls_year = ls_year.droplevel(level = 0)
ls_year = ls_year.divide(1e6)

cd_pur_year = df['cd_pur'].sum(level = [1,1])
cd_pur_year = cd_pur_year.droplevel(level = 0)
cd_pur_year = cd_pur_year.divide(1e6)

cd_sold_year = df['cd_sold'].sum(level = [1,1])
cd_sold_year = cd_sold_year.droplevel(level = 0)
cd_sold_year = cd_sold_year.divide(1e6)

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Loan Sales and Credit Derivatives') 
ax.set(ylabel = 'Total Loan Sale (In $ Billion)', xlabel = 'Year')
ax.plot(ls_year, color = 'black', label = 'Total Loan Sales')

ax2 = ax.twinx()
ax2.set(xlabel='Year', ylabel = 'Credit Derivatives (In $ Billion)')
ax2.plot(cd_pur_year, color = 'black', linestyle = '-.', label = 'Credit Derivatives Purchased')
ax2.plot(cd_sold_year, color = 'black', linestyle = ':', label = 'Credit Derivatives Sold')
ax2.grid(False)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=1)

fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig10a_cdta_lsta.png')

#---------------------------------------------
# Without Loan sales
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Amount of Credit Derivatives') 
ax.set(xlabel='Year', ylabel = 'Credit Derivatives (In $ Billion)')
ax.plot(cd_pur_year, color = 'black', label = 'Credit Derivatives Purchased')
ax.plot(cd_sold_year, color = 'black',linestyle = '-.', label = 'Credit Derivatives Sold')

ax.legend()

fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig10b_cd.png')

#-----------------------------------------------
# Figure 11: Charge-offs 
# Net charge offs
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('On-balance and Total Charge-offs')
ax.set(xlabel='Year', ylabel = 'In $ Billion')
ax.plot((df.RIAD4635.sum(level = [1,1]).droplevel(level = 0).divide(1e6) - \
        df.RIAD4605.sum(level = [1,1]).droplevel(level = 0).divide(1e6)), linestyle = '-',\
        color = 'black', label = 'On-balance Charge-offs')
ax.plot((df[['RIADB747','RIADB748','RIADB749','RIADB750','RIADB751','RIADB752','RIADB753','RIAD4635']].\
        sum(axis = 1).sum(level = [1,1]).droplevel(level = 0).divide(1e6) -\
        df[['RIADB754','RIADB755','RIADB756','RIADB757','RIADB758','RIADB759','RIADB760','RIAD4605']].\
        sum(axis = 1).sum(level = [1,1]).droplevel(level = 0).divide(1e6)), linestyle = '-.',\
        color = 'black', label = 'Total Charge-offs')
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig11_charge_offs.png')

# Net charge offs + allowance rate
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('On-balance and Net Charge-offs and Net Loan Loss Allowances')
ax.set(xlabel='Year', ylabel = 'In %')
ax.plot(df.net_coffratio_tot_ta.mean(level = [1,1]).droplevel(level = 0), linestyle = '-',\
        color = 'black', label = 'Net Charge-offs')
ax.plot(df.allowratio_tot_ta.mean(level = [1,1]).droplevel(level = 0), linestyle = '-.',\
        color = 'black', label = 'Net Loan Loss Allowances')
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()

fig.savefig('Figures\Fig11b_charge_offs_llallow.png')


#-----------------------------------------------
# Figure 12: Total loans + maximum exposure loan sales
## Make variable
tot_credex = df.ls_credex + df.RC2122

#
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Off-balance allowances for credit risk to loan loss allowances')
ax.set(xlabel='Year', ylabel = 'in %')
ax.plot(df[df.RCONB557 > 0.0].RCONB557.sum(level = [1,1]).droplevel(level = 0).divide(df[df.RCONB557 > 0.0].RIAD3123.sum(level = [1,1]).droplevel(level = 0)).multiply(1e2), linestyle = '-',\
        color = 'black')
ax.grid(True)

fig.savefig('Figures\Fig12_off_to_loan_allowances.png')