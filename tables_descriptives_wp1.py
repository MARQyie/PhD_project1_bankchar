#------------------------------------------
# Make tables with descriptives for first working papers
# Mark van der Plaat
# June 2019 

''' 
    This creates tables with summary statistics for working paper #1
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
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style='white',font_scale=1.5)

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0 )

# Make multi index
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Split the dataframes on securitizer and non-securitizers
df_sec = df[df.sec_tot > 0]
df_nonsec = df[df.sec_tot == 0]

#------------------------------------------
# Make function that creates the tables
def makeTables(df,variables,row_labels,column_labels):
    '''This functions creates the summary statistic tables used in this script.'''
    table = pd.DataFrame(df[variables].describe().loc[['mean','50%','std'],:].T.to_numpy(),\
                index = row_labels)

    ## Securitizers
    table = pd.concat([table, pd.DataFrame(df_sec[variables].describe().loc[['mean','50%','std'],:].T.to_numpy(),\
                    index = row_labels)], axis = 1)
    
    ## Non-securitizers
    table = pd.concat([table, pd.DataFrame(df_nonsec[variables].describe().loc[['mean','50%','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)
    
    ## Make columns a multi index
    table.columns = pd.MultiIndex.from_tuples(column_labels)

    return(table)    

#------------------------------------------
# Set column_labels
list_multicolumns = [('Total Sample', 'Mean'), ('Total Sample', 'Median'), ('Total Sample', 'SD'),\
                         ('Securitizers', 'Mean'), ('Securitizers', 'Median'), ('Securitizers', 'SD'),\
                         ('Non-securitizers', 'Mean'), ('Non-securitizers', 'Median'), ('Non-securitizers', 'SD')]

#------------------------------------------
# Table 1: Balance sheet structure
## Set labels and variables
labels_balance = ['Ln Total Assets','Ln Total On- and off Balance Assets',\
                  'Liquidity Ratio','Trading Assets Ratio','Loan Ratio','Loan-to-Deposits',\
                  'Deposit Ratio','Share of Retail Deposits','Capital Ratio','Loan Growth','Asset Growth']
vars_balance = ['size', 'tot_size', 'liqratio', 'traderatio', 'loanratio', 'ltdratio',\
                'depratio', 'retaildep', 'eqratio', 'dloan', 'dass']

## Make table
table_balance = makeTables(df,vars_balance,labels_balance,list_multicolumns)

## Save to Excel
table_balance.to_excel('Table_1_balance_sheet.xlsx')

#------------------------------------------
# Table 2: Loan portfolio
labels_loan = ['Commercial Loan Ratio','Agricultural Loan Ratio']
vars_loan = ['comloanratio', 'agriloanratio']

## Make table
table_loan = makeTables(df,vars_loan,labels_loan,list_multicolumns)

## Save to Excel
table_loan.to_excel('Table_2_loan_portfolio.xlsx')

#------------------------------------------
# Table 3: Regulatory portfolio
labels_regcap = ['Tier 1 Leverage Ratio','Tier 1 Capital Ratio','Regulatory Capital Ratio']
vars_regcap = ['RC7204', 'RC7206', 'RC7205']

## Make table
table_regcap = makeTables(df,vars_regcap,labels_regcap,list_multicolumns)

## Save to Excel
table_regcap.to_excel('Table_3_regulatory_capital.xlsx')

#------------------------------------------
# Table 4: Risk measures
labels_risk = ['RWATA Ratio','Charge-off Ratio','Allowance Ratio','Provision Ratio','Z-score']
vars_risk = ['rwata', 'coffratio', 'allowratio', 'provratio', 'zscore']

## Make table
table_risk = makeTables(df,vars_risk,labels_risk,list_multicolumns)

## Save to Excel
table_risk.to_excel('Table_4_risk_measures.xlsx')

#------------------------------------------
# Table 5: Cost of funding
labels_cost = ['Costs Total Liabilities']
vars_cost = ['intliabratio']

## Make table
table_cost = makeTables(df,vars_cost,labels_cost,list_multicolumns)

## Save to Excel
table_cost.to_excel('Table_5_costs_of_funding.xlsx')

#------------------------------------------
# Table 6: Operatring performance
labels_oper = ['Return on Equity','Return on Assets','Net Interest Margin',\
               'Cost-to-Income Ratio','Non-interest Income/Net Operating Revenue']
vars_oper = ['roe', 'roa', 'nim', 'costinc', 'nonincoperinc']

## Make table
table_oper = makeTables(df,vars_oper,labels_oper,list_multicolumns)

## Save to Excel
table_oper.to_excel('Table_6_operating_performance.xlsx')

#------------------------------------------
# Table 7: Interest income 
labels_intinc = ['Interest Income: Loans','Interest Income: Deposit Institutions',\
                 'Interest Income: Securities','Interest Income: Trading Assets',\
                 'Interest Income: REPOs','Interest Income: Other']
vars_intinc = ['intincloan', 'intincdepins', 'intincsec', 'intinctrade', 'intincrepo', 'intincoth']

## Make table
table_intinc = makeTables(df,vars_intinc,labels_intinc,list_multicolumns)

## Save to Excel
table_intinc.to_excel('Table_7_interest_income.xlsx')

#------------------------------------------
# Table 8: Interest expense
labels_intexp = ['Interest Expenses: REPO','Interest Expenses: Trading Liabilities',\
                 'Interest Expenses: Subordinated Notes']
vars_intexp = ['intexprepo', 'intexptrade','intexpsub']

## Make table
table_intexp = makeTables(df,vars_intexp,labels_intexp,list_multicolumns)

## Save to Excel
table_intexp.to_excel('Table_8_interest_expense.xlsx')

#------------------------------------------
# Table 9: Other
labels_oth = ['Number of Branches']
vars_oth = ['num_branch']

## Make table
table_oth = makeTables(df,vars_oth,labels_oth,list_multicolumns)

## Save to Excel
table_oth.to_excel('Table_9_other.xlsx')