#------------------------------------------
# Make tables with descriptives for first working papers
# Mark van der Plaat
# June 2019 

''' 
    This creates tables with summary statistics for working paper #1
    The tables display the mean, median, standard deviation for the total sample
    securitizers and non-securitizers. 

    Data used: US Call reports 2001-2018 year-end
    
    First run: Data_setup_wp1.py and Add_vars_to_df.py and Clean_data_wp1.py
    ''' 
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
sns.set(style='white',font_scale=1.5)

import os
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

#------------------------------------------
# Load df
df = pd.read_csv('Data\df_wp1_clean.csv', index_col = 0 )

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip())
df.set_index(['IDRSSD','date'],inplace=True)

# Drop missings on distance
df.dropna(subset = ['distance'], inplace = True)

# Dummy variable for loan sales
df['dum_ls'] = (df.ls_tot > 0) * 1

#------------------------------------------
# Make function that creates the tables
def makeTables(df,variables,row_labels,column_labels):
    '''This functions creates the summary statistic tables used in this script.'''
    
    ## Total Sample
    table = pd.DataFrame(df[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                index = row_labels)
    
    ## Split the df
    df_ls = df[df.ls_tot > 0]
    df_nonls = df[df.ls_tot == 0]

    ## Loan Selles
    table = pd.concat([table, pd.DataFrame(df_ls[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)    
    
    ## Non-Sellers
    table = pd.concat([table, pd.DataFrame(df_nonls[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)
    
    ## Difference in means column (absolute value, percentage and t-stat)
    table = pd.concat([table, pd.DataFrame(df_ls[variables].describe().loc[['mean'],:].T.to_numpy(),\
                    index = row_labels) - pd.DataFrame(df_nonls[variables].describe().loc[['mean'],:].T.to_numpy(),\
                    index = row_labels)], axis = 1)        
    table = pd.concat([table, ((pd.DataFrame(df_ls[variables].describe().loc[['mean'],:].T.to_numpy(),\
                    index = row_labels) / pd.DataFrame(df_nonls[variables].describe().loc[['mean'],:].T.to_numpy(),\
                    index = row_labels) - 1) * 100).replace(np.inf, np.nan)], axis = 1)
    
    ### T-stat (Welch Method: unequal size and variance)
    t_test = []
    for i in range(len(variables)):
        stat, pval = ttest_ind(df_ls[variables].iloc[:,i], df_nonls[variables].iloc[:,i],\
                     equal_var = False, nan_policy = 'omit')
        t_test.append(pval)
    
    table = pd.concat([table, pd.DataFrame(t_test, index = row_labels)], axis = 1)
    
    ## Make columns a multi index
    table.columns = pd.MultiIndex.from_tuples(column_labels)

    return(table)    

#------------------------------------------
# Add variables
df['tot_size_exp'] = np.exp(df.tot_size) / 1e6
df['RC2170bln'] = df.RC2170 / 1e6
df['RC2122bln'] = df.RC2122 / 1e6

#------------------------------------------
# Set column_labels
list_multicolumns = [('Total Sample', 'Mean'), ('Total Sample', 'SD'),\
        ('Loan Sellers', 'Mean'), ('Loan Sellers', 'SD'),\
        ('Non-loan Sellers', 'Mean'), ('Non-loan Sellers', 'SD'), \
        ('Difference in Means', 'Abs'),('Difference in Means', '%'),('Difference in Means', 'p-value')]

#------------------------------------------
# Table 1: Variables Call Reports
## Set labels and variables
vars_call = ['RC2170bln', 'tot_size_exp', 'RC2122bln', 'ls_tot_ta','dum_ls','net_coffratio_tot_ta',\
             'allowratio_tot_ta','provratio','RC7204','RC7205','loanratio','roa','nim','nnim','depratio',\
             'comloanratio','mortratio','consloanratio','loanhhi','bhc','RIAD4150']

labels_call = ['Total Assets (\$ bln)','Total Assets (On + Off; \$ bln)', 'Total Loans (\$ bln)', 'Total Loan Sales-to-TA',\
               'Dummy Loan Sales','Total Net Charge-offs-to-TA','Total Allowances-to-TA', 'Provision Ratio',\
               'Regulatory Leverage Ratio', 'Regulatory Capital Ratio', 'Loans-to-TA','Return on Assets',\
               'Net Interest Margin','Net Non-Interest Margin','Deposit Ratio','Commercial Loan Ratio',\
               'Mortgage Ratio','Consumer Loan Ratio','Loan HHI','BHC Indicator','Number of Employees']

vars_sod = ['num_branch','distance','perc_limited_branch','perc_full_branch','unique_states','UNIT']
labels_sod = ['Number of Branches','Maximum Distance Branches','Percentage Limited Service',\
              'Percentage Full Service','Number of States Active', 'Unit Indicator']

## Make table
table_full = makeTables(df,vars_call + vars_sod,labels_call + labels_sod,list_multicolumns)

## Save to Excel
table_full.to_excel('Tables\Summary_statistics_full_sample.xlsx',float_format="%.4f")
table_full.to_latex('Tables\Summary_statistics_full_sample.tex',float_format="%.4f")

#------------------------------------------
#------------------------------------------
# Make similar tables for Size, crisis and Dodd-Frank subsets
#------------------------------------------
#------------------------------------------

def makeTablesSubsets(df,variables,row_labels,column_labels,subset = 'size'):
    '''This functions creates the summary statistic tables used in this script.
    
        NOTE: Always compares the last two groups'''
    
    if subset == 'size':
        # Split the df in subsets 
        ids_small = df[(df.index.get_level_values(1) == pd.Timestamp(2018,12,31)) & (df.RC2170 < 3e5)].index.get_level_values(0).unique().tolist()
        ids_medium = df[(df.index.get_level_values(1) == pd.Timestamp(2018,12,31)) & (df.RC2170.between(3e5,1e6))].index.get_level_values(0).unique().tolist()
        ids_large = df[(df.index.get_level_values(1) == pd.Timestamp(2018,12,31)) & (df.RC2170 > 1e6)].index.get_level_values(0).unique().tolist()
        
        df_small = df[df.index.get_level_values(0).isin(ids_small)]
        df_med = df[df.index.get_level_values(0).isin(ids_medium)]
        df_large = df[df.index.get_level_values(0).isin(ids_large)]
        
        # Small banks
        table = pd.DataFrame(df_small[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                index = row_labels)
    
        # Medium banks
        table = pd.concat([table, pd.DataFrame(df_med[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)  
    
        # Large banks
        table = pd.concat([table, pd.DataFrame(df_large[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)          
        
    elif subset == 'crisis':
        # Split the df
        df_pre = df[df.index.get_level_values(1) <= pd.Timestamp(2006,12,31)]
        df_during = df[(df.index.get_level_values(1) > pd.Timestamp(2006,12,31)) & (df.index.get_level_values(1) < pd.Timestamp(2010,12,30))]
        df_post = df[df.index.get_level_values(1) >= pd.Timestamp(2010,12,31)]
        
        # Pre
        table = pd.DataFrame(df_pre[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                index = row_labels)
    
        # During
        table = pd.concat([table, pd.DataFrame(df_during[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)  
    
        # Crisis
        table = pd.concat([table, pd.DataFrame(df_post[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)   
    elif subset == 'dodd':
        df_pre = df[df.index.get_level_values(1) <= pd.Timestamp(2009,12,31)]
        df_post = df[df.index.get_level_values(1) > pd.Timestamp(2009,12,31)]
        
        # Pre
        table = pd.DataFrame(df_pre[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                index = row_labels)
    
        # During
        table = pd.concat([table, pd.DataFrame(df_post[variables].describe().loc[['mean','std'],:].T.to_numpy(),\
                    index = row_labels) ], axis = 1)  
    else:
        return([])

    ## Difference in means column (absolute value, percentage and t-stat)
    table = pd.concat([table, pd.DataFrame(table.iloc[:,-4].values - table.iloc[:,-2].values,index = row_labels)], axis = 1)        
    table = pd.concat([table, pd.DataFrame((table.iloc[:,-4].values / table.iloc[:,-2]) * 100, index = row_labels).replace(np.inf, np.nan)], axis = 1)
    
    ### T-stat (Welch Method: unequal size and variance)
    if subset == 'size':
        df1 = df_med
        df2 = df_large
    elif subset == 'crisis':
        df1 = df_during
        df2 = df_post
    else:
        df1 = df_pre
        df2 = df_post
        
    t_test = [] 
    for i in variables:
        stat, pval = ttest_ind(df1[i], df2[i],\
                     equal_var = False, nan_policy = 'omit')
        t_test.append(pval)
    
    table = pd.concat([table, pd.DataFrame(t_test, index = row_labels)], axis = 1)
    
    ## Make columns a multi index
    table.columns = pd.MultiIndex.from_tuples(column_labels)

    return(table) 

# Make the tables
## Prelims
vars_sub = vars_call + vars_sod
labels_sub = labels_call + labels_sod

list_multicolumns_size = [('Small Banks', 'Mean'), ('Small Banks', 'SD'),\
        ('Medium Banks', 'Mean'), ('Medium Banks', 'SD'),\
        ('Large Banks', 'Mean'), ('Large Banks', 'SD'), \
        ('Difference in Means', 'Abs'),('Difference in Means', '%'),('Difference in Means', 'p-value')]
list_multicolumns_crisis = [('Pre-Crisis', 'Mean'), ('Pre-Crisis', 'SD'),\
        ('Crisis', 'Mean'), ('Crisis', 'SD'),\
        ('Post-Crisis', 'Mean'), ('Post-Crisis', 'SD'), \
        ('Difference in Means', 'Abs'),('Difference in Means', '%'),('Difference in Means', 'p-value')]
list_multicolumns_dodd = [('Pre-Dodd-Frank', 'Mean'), ('Pre-Dodd-Frank', 'SD'),\
        ('Post-Dodd-Frank', 'Mean'), ('Post-Dodd-Frank', 'SD'),\
        ('Difference in Means', 'Abs'),('Difference in Means', '%'),('Difference in Means', 'p-value')]

## Tables
table_size = makeTablesSubsets(df,vars_sub,labels_sub,list_multicolumns_size,'size')
table_crisis = makeTablesSubsets(df,vars_sub,labels_sub,list_multicolumns_crisis,'crisis')
table_dodd = makeTablesSubsets(df,vars_sub,labels_sub,list_multicolumns_dodd,'dodd')

## To Excel
with pd.ExcelWriter('Tables\Summary_statistics_subsets.xlsx') as writer:
    table_size.to_excel(writer, sheet_name = 'Size',float_format="%.4f")
    table_crisis.to_excel(writer, sheet_name = 'Crisis',float_format="%.4f")
    table_dodd.to_excel(writer, sheet_name = 'Dodd-Frank',float_format="%.4f")
    
## To Latex
table_size.to_latex('Tables\Summary_statistics_size_sample.tex',float_format="%.4f")
table_crisis.to_latex('Tables\Summary_statistics_crisis_sample.tex',float_format="%.4f")
table_dodd.to_latex('Tables\Summary_statistics_dodd_sample.tex',float_format="%.4f")
