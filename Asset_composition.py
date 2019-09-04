#------------------------------------------
# Asset composition of US banks
# Mark van der Plaat
# June 2019 

''' This document looks at the asset composition of US banks
    There are reasons to believe that securitizing banks have more corporate 
    assets on their balance sheets, which could be a reason why these banks
    securitize more.
    
    Banks will be split on: 1) being a securitizer or not and 2) being a 
    SIFI (TA > $50 bil USD) or not. 
    
    Derivatives and loan sales are included to check whether the use of these
    differ across the groups.
    
    Data used: US Call reports 2001-2018 year-end
    Only insured, commercial banks are taken
    ''' 
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style='white',font_scale=1.5)

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

#------------------------------------------
# Set Prelims 
start = 2001
end = 2019
path_info = r'X:/My Documents/Data/Data_call_reports_fed'
path_call = r'X:/My Documents/Data/Data_call_reports_FFIEC2'
first_run = False

#------------------------------------------
# Set filenames
file_info1 = r'/call0{}12.xpt'
file_info2 = r'/call{}12.xpt'
file_rc = r'/{}/FFIEC CDR Call Schedule RC 1231{}.txt'
file_rcb1 = r'/{}/FFIEC CDR Call Schedule RCB 1231{}.txt'
file_rcb2 = r'/{}/FFIEC CDR Call Schedule RCB 1231{}(1 of 2).txt'
file_rcc = r'/{}/FFIEC CDR Call Schedule RCCI 1231{}.txt'
file_rcl1 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}.txt'
file_rcl2 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(1 of 2).txt'
file_rcl3 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(2 of 2).txt'
file_rcr1_rcfd = r'/{}/FFIEC CDR Call Schedule RCR 1231{}(1 of 2).txt'
file_rcr2_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 2).txt' #2014
file_rcr3_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 3).txt' #from 2015
file_rcr4_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 4).txt' #from 2017
file_rcr1_rcon = r'/{}/FFIEC CDR Call Schedule RCR 1231{}(2 of 2).txt'
file_rcr2_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 2).txt' #2014
file_rcr3_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 3).txt' #from 2015
file_rcr4_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 4).txt' #from 2017
file_rcs = r'/{}/FFIEC CDR Call Schedule RCS 1231{}.txt'
file_ri = r'/{}/FFIEC CDR Call Schedule RI 1231{}.txt'

#------------------------------------------
# Set variables needed per csv
## From balance sheet
vars_info = ['RSSD9001','RSSD9999','RSSD9048','RSSD9424','RSSD9170'] 
#IDRSSD, date, charter type, insured, federal district 
vars_rc = ['IDRSSD','RCFD2170','RCON2170','RCFD3545','RCON3545','RCFD3548','RCON3548','RCON2200'] 
#..., TA, trading assets, trading liabilities (do not fill the NANs for the last two), total deposits
vars_rcb = ['IDRSSD','RCFD1771','RCON1771','RCFD1773','RCON1773','RCFD0213','RCON0213','RCFD1287','RCON1287']
#..., total held-to-maturity securities (fair value), total available-for-sale securities (fair value), total held-to-maturity T-bills (fair value), total available-for-sale T-bills (fair value)
vars_rcc = ['IDRSSD','RCFD1410','RCON1420','RCON1797','RCON1460','RCON1288','RCON1590','RCON1766','RCONB538','RCONB539'] 
#...,total loans, agri mortgages, residential mortgages, multi-family residencial mortgages, loans to other dep. institutions, agri loans, com & ind loans, rest is loans to individuals
vars_rcl = ['IDRSSD','RCFD8725','RCFD8726','RCFD8727','RCFD8728','RCFDA126','RCFDA127','RCFD8723','RCFD8724',\
           'RCON8725','RCON8726','RCON8727','RCON8728','RCONA126','RCONA127','RCON8723','RCON8724'] 
#..., trading and non-trading derivatives for all types
vars_rcl_cd = ['IDRSSD', 'RCFDC968','RCFDC969','RCFDC970','RCFDC971','RCFDC972','RCFDC973','RCFDC974','RCFDC975',\
           'RCONC968','RCONC969','RCONC970','RCONC971','RCONC972','RCONC973','RCONC974','RCONC975'] 
#..., credit derivatives
vars_rcr = ['IDRSSD', 'RCFDB704','RCFDA222','RCFD3128', 'RCONB704','RCONA222','RCON3128']
#..., Total RWA (before deductions), deductions
vars_rcs = ['IDRSSD','RCFDB705','RCFDB711','RCONB705','RCONB711','RCFDB790','RCFDB796','RCONB790','RCONB796'] 
#..., securitizations + loan sales (with recourse)
vars_ri = ['IDRSSD', 'RIAD4074','RIAD4230','RIAD4079','RIAD4080']
#..., net interest income, loan loss provisions, non-interest income, deposit fees

#------------------------------------------
# Load the data
if first_run:
    ## define the dfs
    df_info = pd.DataFrame()
    df_rc = pd.DataFrame()
    df_rcb = pd.DataFrame()
    df_rcc = pd.DataFrame()
    df_rcl = pd.DataFrame()
    df_rcl_cd = pd.DataFrame()
    df_rcr = pd.DataFrame()
    df_rcs = pd.DataFrame()
    df_ri = pd.DataFrame()
    
    ## Load info data
    for i in range(start - 2000, end - 2000):
        if i < 10:
            df_load = pd.read_sas((path_info + file_info1).format(i))
        else:
            df_load = pd.read_sas((path_info + file_info2).format(i))
        df_load = df_load[vars_info]
        df_info = df_info.append(df_load)
    
    ### Rename RSSD9001 and RSSD9999
    df_info.rename(columns = {'RSSD9001':'IDRSSD', 'RSSD9999':'date'}, inplace = True)
    
    ### Change date to only the year
    df_info.date = (df_info.date.round(-4) / 1e4).astype(int)
    
    ## Load rc data
    for i in range(start,end):
        df_load = pd.read_csv((path_call + file_rc).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load[vars_rc]
        df_load['date'] = int('{}'.format(i))
        df_rc = df_rc.append(df_load)
        
    ### Merge RCFD and RCON cols
    var_num = ['2170', '3545', '3548']
    
    for i,elem in enumerate(var_num):
        '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
        df_rc['RC{}'.format(elem)] = df_rc.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1)      
    
    ## Load rcb data
    for i in range(start,end):
        if i < 2009:
            df_load = pd.read_csv((path_call + file_rcb1).format(i,i), \
                     sep='\t',  skiprows = [1,2])
        else:
            df_load = pd.read_csv((path_call + file_rcb2).format(i,i), \
                     sep='\t',  skiprows = [1,2])
        df_load = df_load[vars_rcb]
        df_load['date'] = int('{}'.format(i))
        df_rcb = df_rcb.append(df_load)  
    
    ### Merge RCFD and RCON cols    
    var_num = ['1771','1773','0213','1287']
  
    for i,elem in enumerate(var_num):
        '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
        df_rcb['RC{}'.format(elem)] = df_rcb.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1)      
    
    ## Load rcc data
    for i in range(start,end):
        df_load = pd.read_csv((path_call + file_rcc).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load[vars_rcc]
        df_load['date'] = int('{}'.format(i))
        df_rcc = df_rcc.append(df_load) 
        
    ## Load rcl data  
    for i in range(start,end):
        if i < 2009:
            df_load = pd.read_csv((path_call + file_rcl1).format(i,i), \
                     sep='\t',  skiprows = [1,2])
        else:
            df_load = pd.read_csv((path_call + file_rcl2).format(i,i), \
                     sep='\t',  skiprows = [1,2])
        df_load = df_load[vars_rcl]
        df_load['date'] = int('{}'.format(i))
        df_rcl = df_rcl.append(df_load)     
    
    ### Merge RCFD and RCON cols    
    var_num = ['8725','8726','8727','8728','A126','A127','8723','8724']
    
    for i,elem in enumerate(var_num):
        '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
        df_rcl['RC{}'.format(elem)] = df_rcl.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1)    
    
    ## load rcl_cd data
    for i in range(start + 5,end):
        if i < 2009:
            df_load = pd.read_csv((path_call + file_rcl1).format(i,i), \
                     sep='\t',  skiprows = [1,2])
            df_load = df_load[vars_rcl_cd]
            df_load['date'] = int('{}'.format(i))
        else:
            df_load = pd.read_csv((path_call + file_rcl2).format(i,i), \
                     sep='\t',  skiprows = [1,2])
            df_load2 = pd.read_csv((path_call + file_rcl3).format(i,i), \
                     sep='\t',  skiprows = [1,2])
            df_load['date'], df_load2['date'] = int('{}'.format(i)), int('{}'.format(i))
            df_load = df_load.merge(df_load2, on = ['IDRSSD', 'date'], how = 'outer')
            df_load = df_load[vars_rcl_cd]
            df_load['date'] = int('{}'.format(i))
    
        df_rcl_cd = df_rcl_cd.append(df_load)
    
    ### Merge RCFD and RCON cols    
    var_num = ['C968','C969','C970','C971','C972','C973','C974','C975']
    
    for i,elem in enumerate(var_num):
        '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
        df_rcl_cd['RC{}'.format(elem)] = df_rcl_cd.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1) 
    
    ## Load rcr data
    for i in range(start,end):
        if i == 2014:
            df_load_rcfd = pd.read_csv((path_call + file_rcr2_rcfd).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
            df_load_rcon = pd.read_csv((path_call + file_rcr2_rcon).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
        elif i == 2015 or i == 2016:
            df_load_rcfd = pd.read_csv((path_call + file_rcr3_rcfd).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
            df_load_rcon = pd.read_csv((path_call + file_rcr3_rcon).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
        elif i > 2016:
            df_load_rcfd = pd.read_csv((path_call + file_rcr4_rcfd).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
            df_load_rcon = pd.read_csv((path_call + file_rcr4_rcon).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
        else:
            df_load_rcfd = pd.read_csv((path_call + file_rcr1_rcfd).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
            df_load_rcon = pd.read_csv((path_call + file_rcr1_rcon).format(i,i), \
                     sep='\t',  skiprows = [1,2]) 
        
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load[vars_rcr]
        df_load['date'] = int('{}'.format(i))
        
        df_rcr = df_rcr.append(df_load)
     
    ### Make Variables G641 (total RWA) and merge the two
    df_rcr['RCFDG641'] = df_rcr.RCFDB704 - df_rcr.RCFDA222 - df_rcr.RCFD3128
    df_rcr['RCONG641'] = df_rcr.RCONB704 - df_rcr.RCONA222 - df_rcr.RCON3128
    
    var_num = ['G641']
    
    for i,elem in enumerate(var_num):
        '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
        df_rcr['RC{}'.format(elem)] = df_rcr.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1) 

    ## Load rcs data    
    for i in range(start,end):
        df_load = pd.read_csv((path_call + file_rcs).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load[vars_rcs]
        df_load['date'] = int('{}'.format(i))
        df_rcs = df_rcs.append(df_load) 
    
    ### Merge RCFD and RCON cols
    var_num = ['B705','B711','B790','B796']
    
    for i,elem in enumerate(var_num):
        '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
        df_rcs['RC{}'.format(elem)] = df_rcs.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1) 
        
    df_rcs['RCBtot'] = df_rcs.apply(lambda x: x.RCB705 + x.RCB711 + x.RCB790 + x.RCB796, axis = 1) 
    
    ## Load ri data    
    for i in range(start,end):
        df_load = pd.read_csv((path_call + file_ri).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load[vars_ri]
        df_load['date'] = int('{}'.format(i))
        df_ri = df_ri.append(df_load) 

    #------------------------------------------
    # Merge data and save the raw file 
    df_raw = df_rc.merge(df_rcc, on = ['IDRSSD', 'date'], how = 'outer')
    df_raw = df_raw.merge(df_rcb, on = ['IDRSSD', 'date'], how = 'outer')
    df_raw = df_raw.merge(df_rcl, on = ['IDRSSD', 'date'], how = 'outer')
    df_raw = df_raw.merge(df_rcl_cd, on = ['IDRSSD', 'date'], how = 'left')
    df_raw = df_raw.merge(df_rcr, on = ['IDRSSD', 'date'], how = 'left')
    df_raw = df_raw.merge(df_rcs, on = ['IDRSSD', 'date'], how = 'outer')
    df_raw = df_raw.merge(df_ri, on = ['IDRSSD', 'date'], how = 'left')
    df_raw = df_raw.merge(df_info, on = ['IDRSSD', 'date'], how = 'left')
    
    df_raw.to_csv('df_assetcomp_raw.csv')

    #------------------------------------------
    # Clean data and save
    ## Select which columns to use
    
    cols_df = ['IDRSSD','date','RC2170','RC3545','RC3548','RCON2200', 'RC1771', 'RC1773',\
               'RC0213', 'RC1287','RCFD1410', 'RCON1420', 'RCON1797', 'RCON1460', 'RCON1288', \
               'RCON1590', 'RCON1766', 'RCONB538', 'RCONB539', \
               'RC8725', 'RC8726', 'RC8727', 'RC8728', 'RCA126', 'RCA127', 'RC8723', 'RC8724',\
               'RCC968', 'RCC969', 'RCC970', 'RCC971', 'RCC972', 'RCC973', 'RCC974', 'RCC975', 'RCG641',\
               'RCB705','RCB711', 'RCB790', 'RCB796', 'RCBtot', 'RSSD9048', 'RSSD9424','RSSD9170',\
               'RIAD4074','RIAD4230','RIAD4079','RIAD4080']    
    
    ## Make new dataset
    df = df_raw[cols_df]
    
    ## Select insured, commercial banks
    '''RSSD9048 == 200, RSSD9424 != 0'''
    df = df[(df.RSSD9048 == 200) & (df.RSSD9424 != 0)]
    
    ## Fill and drop nans
    cols_nanfill = ['RCON1288','RC8725', 'RC8726', 'RC8727','RC8728', 'RCA126', 'RCA127', 'RC8723', 'RC8724',\
                    'RCB705', 'RCB711', 'RCB790','RCB796', 'RCBtot','RIAD4074','RIAD4230','RIAD4079','RIAD4080']
    cols_nandrop = ['RC2170','RCON2200','RCON1590','RCON1766','RCONB538','RCONB539','RCG641']
    
    df[cols_nanfill] = df[cols_nanfill].fillna(value = 0)
    df.dropna(subset = cols_nandrop,inplace = True)
    
    ## Only accept positive total assets
    df = df[df.RC2170 > 0.0]
    df = df[df.RCG641 > 0.0]
    
    ## Save df
    df.to_csv('df_assetcomp.csv')

#------------------------------------------
# Load df
else:
    df = pd.read_csv('df_assetcomp.csv', index_col = 0)    

#------------------------------------------
# Analysis
df['hhloans'] = df[['RCONB538', 'RCONB539']].sum(axis = 1) 
df['CD_sold'] = df[['RCC968', 'RCC970', 'RCC972', 'RCC974']].sum(axis = 1) 
df['CD_pur'] = df[['RCC969', 'RCC971', 'RCC973', 'RCC975']].sum(axis = 1)
df['sec_tot'] = df[['RCB705', 'RCB711']].sum(axis = 1)
df['ls_tot'] = df[['RCB790', 'RCB796']].sum(axis = 1)
df['TA_RIAD'] = df.RC2170 * (1 + ((df.RIAD4079 - df.RIAD4080)/(df.RIAD4074 - df.RIAD4230)))  
    
vars_desc = ['RC2170', 'RC3545', 'RC3548', 'RCON2200', 'RC1771', 'RC1773', 'RC0213', 'RC1287', 'RCFD1410', 'RCON1420', \
             'RCON1797', 'RCON1460', 'RCON1288', 'RCON1590', 'RCON1766', 'hhloans' ,'RC8725', 'RC8726', 'RC8727',\
             'RC8728', 'RCA126', 'RCA127', 'RC8723', 'RC8724', 'CD_sold' , 'CD_pur', 'sec_tot', 'ls_tot']
vars_desc_names = ['Total Assets', 'Trading Assets', 'Trading Liabilities', 'Total Deposits', 'Securities (HtM)', \
                   'Securities (AfS)', 'T-bills (HtM)', 'T-bills (AfS)', 'Total Mortgages', 'Farm Mortgages', \
                   'Home Mortgages', 'Multi. Fam. Mortgages', 'Loans to Dep. Inst.', 'Farm Loans', 'Commercial Loans', \
                   'Household Loans', 'IRDs (Hedge)', 'FExDs (Hedge)', 'EqDs (Hedge)', 'CoDs (Hedge)', 'IRDs (Trade)', \
                   'FExDs (Trade)', 'EqDs (Trade)', 'CoDs (Trade)', 'Credit Der. (Sold)', 'Credit Der. (Purchased)', \
                   'Total Securitization', 'Total Loan Sales']

## Make descriptives table for the total sample 
table_1 = pd.DataFrame(df[vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)    
    
## Split sample on crisis and add to table
pre_crisis = range(2001,2007)
crisis = range(2007,2010)
post_crisis = range(2010,2019)

table_1 = pd.concat([table_1, pd.DataFrame(df[(df.date.isin(pre_crisis))][vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)], axis = 1)

table_1 = pd.concat([table_1, pd.DataFrame(df[(df.date.isin(crisis))][vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)], axis = 1)

table_1 = pd.concat([table_1, pd.DataFrame(df[(df.date.isin(post_crisis))][vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)], axis = 1)

### Number of obs
nobs = df.shape[0] #113631
nobs_pre_crisis = df[(df.date.isin(pre_crisis))].shape[0] #44298
nobs_crisis = df[(df.date.isin(crisis))].shape[0] #20451
nobs_post_crisis = df[(df.date.isin(post_crisis))].shape[0] #48882

nobs_list = [nobs, nobs, nobs_pre_crisis, nobs_pre_crisis, nobs_crisis, nobs_crisis, nobs_post_crisis, \
                      nobs_post_crisis]
table_1.loc[-1] = nobs_list
table_1.rename({-1:'N. Obs.'}, axis = 'index', inplace = True)

### Save table 
columns = [('Total Sample', 'Mean'), ('Total Sample', 'Median'), ('Pre-crisis', 'Mean'), ('Pre-crisis', 'Median'), \
           ('Crisis', 'Mean'), ('Crisis', 'Median'), ('Post-crisis', 'Mean'), ('Post-crisis', 'Median')]
table_1.columns = pd.MultiIndex.from_tuples(columns)

table_1.to_excel('table_1.xlsx') 

## Do the same for large vs small banks (split at 3 billion TA) and securitizers vs non
table_2 = pd.DataFrame(df[vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)  
table_2 = pd.concat([table_2, pd.DataFrame(df[df.RC2170 < 3e7][vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)], axis = 1)
table_2 = pd.concat([table_2, pd.DataFrame(df[df.RC2170 >= 3e7][vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)], axis = 1)
table_2 = pd.concat([table_2, pd.DataFrame(df[(df.sec_tot) == 0][vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)], axis = 1)
table_2 = pd.concat([table_2, pd.DataFrame(df[(df.sec_tot) > 0][vars_desc].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names)], axis = 1)

nobs_list = [nobs, nobs, df[df.RC2170 < 3e7].shape[0], df[df.RC2170 < 3e7].shape[0], df[df.RC2170 >= 3e7].shape[0], \
             df[df.RC2170 >= 3e7].shape[0], df[(df.sec_tot) == 0].shape[0], df[(df.sec_tot) == 0].shape[0], \
             df[(df.sec_tot) > 0].shape[0], df[(df.sec_tot) > 0].shape[0]]
table_2.loc[-1] = nobs_list
table_2.rename({-1:'N. Obs.'}, axis = 'index', inplace = True)

### Save table 
columns = [('Total Sample', 'Mean'), ('Total Sample', 'Median'), ('Small Banks', 'Mean'), ('Small Banks', 'Median'), \
           ('Large Banks', 'Mean'), ('Large Banks', 'Median'), ('Non-securitizing Banks', 'Mean'), \
           ('Non-securitizing Banks', 'Median'), ('Securitizing Banks', 'Mean'), ('Securitizing Banks', 'Median')]
table_2.columns = pd.MultiIndex.from_tuples(columns)

table_2.to_excel('table_2.xlsx') 

## Do the same, but now relative to TA
df2 = df.iloc[:,3:].div(df.RC2170, axis = 0) * 100
df2 = pd.concat([df.iloc[:,:3],df2], axis = 1)

vars_desc2 = ['RC3545', 'RC3548', 'RCON2200', 'RC1771', 'RC1773', 'RC0213', 'RC1287', 'RCFD1410', 'RCON1420', \
             'RCON1797', 'RCON1460', 'RCON1288', 'RCON1590', 'RCON1766', 'hhloans' ,'RC8725', 'RC8726', 'RC8727',\
             'RC8728', 'RCA126', 'RCA127', 'RC8723', 'RC8724', 'CD_sold' , 'CD_pur', 'sec_tot', 'ls_tot']
vars_desc_names2 = ['Trading Assets', 'Trading Liabilities', 'Total Deposits', 'Securities (HtM)', \
                   'Securities (AfS)', 'T-bills (HtM)', 'T-bills (AfS)', 'Total Mortgages', 'Farm Mortgages', \
                   'Home Mortgages', 'Multi. Fam. Mortgages', 'Loans to Dep. Inst.', 'Farm Loans', 'Commercial Loans', \
                   'Household Loans', 'IRDs (Hedge)', 'FExDs (Hedge)', 'EqDs (Hedge)', 'CoDs (Hedge)', 'IRDs (Trade)', \
                   'FExDs (Trade)', 'EqDs (Trade)', 'CoDs (Trade)', 'Credit Der. (Sold)', 'Credit Der. (Purchased)', \
                   'Total Securitization', 'Total Loan Sales']

### Table 3
table_3 = pd.DataFrame(df2[vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)    
table_3 = pd.concat([table_3, pd.DataFrame(df2[(df2.date.isin(pre_crisis))][vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)], axis = 1)
table_3 = pd.concat([table_3, pd.DataFrame(df2[(df2.date.isin(crisis))][vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)], axis = 1)
table_3 = pd.concat([table_3, pd.DataFrame(df2[(df2.date.isin(post_crisis))][vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)], axis = 1)

nobs_list = [nobs, nobs, nobs_pre_crisis, nobs_pre_crisis, nobs_crisis, nobs_crisis, nobs_post_crisis, \
                      nobs_post_crisis]
table_3.loc[-1] = nobs_list
table_3.rename({-1:'N. Obs.'}, axis = 'index', inplace = True)

columns = [('Total Sample (% of TA)', 'Mean'), ('Total Sample (% of TA)', 'Median'), ('Pre-crisis (% of TA)', 'Mean'), ('Pre-crisis (% of TA)', 'Median'), \
           ('Crisis (% of TA)', 'Mean'), ('Crisis (% of TA)', 'Median'), ('Post-crisis (% of TA)', 'Mean'), ('Post-crisis (% of TA)', 'Median')]
table_3.columns = pd.MultiIndex.from_tuples(columns)

table_3.to_excel('table_3.xlsx') 

### Table 4
table_4 = pd.DataFrame(df2[vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)  
table_4 = pd.concat([table_4, pd.DataFrame(df2[df2.RC2170 < 3e7][vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)], axis = 1)
table_4 = pd.concat([table_4, pd.DataFrame(df2[df2.RC2170 >= 3e7][vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)], axis = 1)
table_4 = pd.concat([table_4, pd.DataFrame(df2[(df2.sec_tot) == 0][vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)], axis = 1)
table_4 = pd.concat([table_4, pd.DataFrame(df2[(df2.sec_tot) > 0][vars_desc2].describe().loc[['mean','50%'],:].T.to_numpy(), index = vars_desc_names2)], axis = 1)

nobs_list = [nobs, nobs, df[df.RC2170 < 3e7].shape[0], df[df.RC2170 < 3e7].shape[0], df[df.RC2170 >= 3e7].shape[0], \
             df[df.RC2170 >= 3e7].shape[0], df[(df.sec_tot) == 0].shape[0], df[(df.sec_tot) == 0].shape[0], \
             df[(df.sec_tot) > 0].shape[0], df[(df.sec_tot) > 0].shape[0]]
table_4.loc[-1] = nobs_list
table_4.rename({-1:'N. Obs.'}, axis = 'index', inplace = True)

### Save table 
columns = [('Total Sample (% of TA)', 'Mean'), ('Total Sample (% of TA)', 'Median'), ('Small Banks (% of TA)', 'Mean'), ('Small Banks (% of TA)', 'Median'), \
           ('Large Banks (% of TA)', 'Mean'), ('Large Banks (% of TA)', 'Median'), ('Non-securitizing Banks (% of TA)', 'Mean'), \
           ('Non-securitizing Banks (% of TA)', 'Median'), ('Securitizing Banks (% of TA)', 'Mean'), ('Securitizing Banks (% of TA)', 'Median')]
table_4.columns = pd.MultiIndex.from_tuples(columns)

table_4.to_excel('table_4.xlsx') 

#------------------------------------------
# Figures
## Histogram securitization 
num_bins = np.histogram(np.hstack((df.RCB705,df.RCB711)), bins=25)[1]
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Historgram of the amount of securitization')
ax.set_ylabel('Count')
ax.set(xlabel='Bins')
ax.hist((df.RCB705, df.RCB711), 25, stacked = True, label = ('Residential loans', 'Other'),\
        log = True)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(df)))
plt.legend(loc='upper right')
plt.show()

fig.savefig('histogram_RCB705711.png')

## Figure (bar) securitization (B705/B711) and size
### Prelims
df['lnRC2170'] = np.log(df.RC2170)
df_subset = df[(df.date == 2016) | (df.date == 2017) | (df.date == 2018)]
df_subset['bins_size'], bins = pd.cut(df_subset.lnRC2170, 25, retbins = True)
df_mean = df_subset.groupby(['date','bins_size']).RCB705.sum()

df_count = df_subset.groupby(['date','bins_size']).RCB705.count()
df_count = np.cumsum(df_count.unstack(level = 0).fillna(0)) / df_count.unstack(level = 0).fillna(0).sum()

ind = np.arange(df_mean.unstack(level = 0).iloc[:,0].shape[0])
width = 1/3
colors = ['white','grey','black']
hatches = ['\\','o','-']
lines = ['-','-.',':']

### Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitization of Residential Loans (By Bank-Size)')
ax.set_ylabel('Amount of securitization (in $1000); bars')
ax.set(xlabel='Bank size (in mln $)')
for i in range(3):
    ax.bar([j + width * i for j in ind], df_mean.unstack(level = 0).iloc[:,i], width, label = '{}'.format(2016 + i), \
           log = True, color = colors[i], edgecolor = 'black', hatch = hatches[i])
ax2 = ax.twinx()
ax2.set_ylabel('Percentage of banks; line')
for k in range(3):
    ax2.plot(ind, df_count.iloc[:,k], color = 'black', linestyle = lines[k])
ax2.axhline(0.5, alpha = 0.5, linestyle = '--', color = 'black')
ax2.text(x = ind[-1], y = 0.51, s = '50%', alpha = 0.5)
ax2.axhline(0.95, alpha = 0.5, linestyle = ':', color = 'black')
ax2.text(x = ind[-1], y = 0.96, s = '95%', alpha = 0.5)
ax.set_xticks([j + width * 1 for j in ind])
ax.set_xticklabels(np.round(np.exp(bins) / 1000, 2), rotation='vertical') # in millions
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=2)
plt.tight_layout()
plt.show()

fig.savefig('sec_banksize.png')

## Plot for all years 
### Prelim
df_mean2 = df_subset.groupby(['bins_size']).RCB705.sum()

ind2 = np.arange(df_mean2.shape[0])
width = 1

###PLOT
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Securitization of Residential Loans (By Bank-Size)')
ax.set_ylabel('Amount of securitization (in $1000)')
ax.set(xlabel='Bank size (in mln $)')
ax.bar(ind2, df_mean2, width, log = True, color = 'grey', edgecolor = 'black')
ax.set_xticks([j + width * 1 for j in ind2])
ax.set_xticklabels(np.round(np.exp(bins) / 1000, 2), rotation='vertical') # in millions
plt.tight_layout()
plt.show()

fig.savefig('sec_banksize_all.png')


## Loan sales plot
### Prelims
df_mean3 = df_subset.groupby(['date','bins_size']).RCB790.sum()

ind = np.arange(df_mean.unstack(level = 0).iloc[:,0].shape[0])
width = 1/3
colors = ['white','grey','black']
hatches = ['\\','o','-']
lines = ['-','-.',':']

### Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Loans Sales with Recourse (By Bank-Size)')
ax.set_ylabel('Amount of Loan Sales (in $1000); bars')
ax.set(xlabel='Bank Size (in mln $)')
for i in range(3):
    ax.bar([j + width * i for j in ind], df_mean3.unstack(level = 0).iloc[:,i], width, label = '{}'.format(2016 + i), \
           log = True, color = colors[i], edgecolor = 'black', hatch = hatches[i])
ax2 = ax.twinx()
ax2.set_ylabel('Percentage of banks; line')
for k in range(3):
    ax2.plot(ind, df_count.iloc[:,k], color = 'black', linestyle = lines[k])
ax2.axhline(0.5, alpha = 0.5, linestyle = '--', color = 'black')
ax2.text(x = ind[-1], y = 0.51, s = '50%', alpha = 0.5)
ax2.axhline(0.95, alpha = 0.5, linestyle = ':', color = 'black')
ax2.text(x = ind[-1], y = 0.96, s = '95%', alpha = 0.5)
ax.set_xticks([j + width * 1 for j in ind])
ax.set_xticklabels(np.round(np.exp(bins) / 1000, 2), rotation='vertical') # in millions
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=2)
plt.tight_layout()
plt.show()

fig.savefig('ls_banksize.png')

## securities
### Prelims
df_mean4 = df_subset.groupby(['date','bins_size']).RC1771.mean() / df_subset.groupby(['date','bins_size']).RC2170.mean()
df_mean5 = df_subset.groupby(['date','bins_size']).RC1773.mean() / df_subset.groupby(['date','bins_size']).RC2170.mean()

ind = np.arange(df_mean.unstack(level = 0).iloc[:,0].shape[0])
width = 1/3
colors1 = ['white','white','white']
colors2 = ['grey','grey','grey']
hatches = ['\\','o','-']
lines = ['-','-.',':']

### Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Percentage of securities on the balance sheet')
ax.set_ylabel('Percentage; bars')
ax.set(xlabel='Bank Size (in mln $)')
for i in range(3):
    ax.bar([j + width * i for j in ind], df_mean4.unstack(level = 0).iloc[:,i], width, label = '{} (HtM)'.format(2016 + i), \
           log = True, color = colors1[i], edgecolor = 'black', hatch = hatches[i])
    ax.bar([j + width * i for j in ind], df_mean5.unstack(level = 0).iloc[:,i], width, label = '{} (AfS)'.format(2016 + i), \
           log = True, color = colors2[i], edgecolor = 'black', hatch = hatches[i], bottom = df_mean4.unstack(level = 0).iloc[:,i])
ax2 = ax.twinx()
ax2.set_ylabel('Percentage of banks; line')
for k in range(3):
    ax2.plot(ind, df_count.iloc[:,k], color = 'black', linestyle = lines[k])
ax2.axhline(0.5, alpha = 0.5, linestyle = '--', color = 'black')
ax2.text(x = ind[-1], y = 0.51, s = '50%', alpha = 0.5)
ax2.axhline(0.95, alpha = 0.5, linestyle = ':', color = 'black')
ax2.text(x = ind[-1], y = 0.96, s = '95%', alpha = 0.5)
ax.set_xticks([j + width * 1 for j in ind])
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.set_xticklabels(np.round(np.exp(bins) / 1000, 2), rotation='vertical') # in millions
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=1)
plt.tight_layout()
plt.show()

fig.savefig('securities_banksize.png')

## Figure (bar) securitization (B705/B711) and size (RWA)
### Prelims
df['lnRCG641'] = np.log(df.RCG641)
df_subset = df[(df.date == 2016) | (df.date == 2017) | (df.date == 2018)]
df_subset['bins_size_rwa'], bins = pd.cut(df_subset.lnRCG641, 25, retbins = True)
df_mean6 = df_subset.groupby(['date','bins_size_rwa']).RCB705.sum()

df_count = df_subset.groupby(['date','bins_size_rwa']).RCB705.count()
df_count = np.cumsum(df_count.unstack(level = 0).fillna(0)) / df_count.unstack(level = 0).fillna(0).sum()

ind = np.arange(df_mean.unstack(level = 0).iloc[:,0].shape[0])
width = 1/3
colors = ['white','grey','black']
hatches = ['\\','o','-']
lines = ['-','-.',':']

### Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitization of Residential Loans (By Bank-Size; RWA)')
ax.set_ylabel('Amount of securitization (in $1000; bars')
ax.set(xlabel='Bank size (in mln $)')
for i in range(3):
    ax.bar([j + width * i for j in ind], df_mean6.unstack(level = 0).iloc[:,i], width, label = '{}'.format(2016 + i), \
           log = True, color = colors[i], edgecolor = 'black', hatch = hatches[i])
ax2 = ax.twinx()
ax2.set_ylabel('Percentage of banks; line')
for k in range(3):
    ax2.plot(ind, df_count.iloc[:,k], color = 'black', linestyle = lines[k])
ax2.axhline(0.5, alpha = 0.5, linestyle = '--', color = 'black')
ax2.text(x = ind[-1], y = 0.51, s = '50%', alpha = 0.5)
ax2.axhline(0.95, alpha = 0.5, linestyle = ':', color = 'black')
ax2.text(x = ind[-1], y = 0.96, s = '95%', alpha = 0.5)
ax.set_xticks([j + width * 1 for j in ind])
ax.set_xticklabels(np.round(np.exp(bins) / 1000, 2), rotation='vertical') # in millions
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=2)
plt.tight_layout()
plt.show()

fig.savefig('sec_banksize_rwa.png')

## Figure (bar) securitization (B705/B711) and size (Income approach)
### Prelims
df['lnTA_RIAD'] = np.log(df.TA_RIAD)
df = df[~(df.lnTA_RIAD == np.infty) & ~(df.lnTA_RIAD == -np.infty)]
df_subset = df[(df.date == 2016) | (df.date == 2017) | (df.date == 2018)]
df_subset['bins_size_riad'], bins = pd.cut(df_subset.lnTA_RIAD, 25, retbins = True)
df_mean6 = df_subset.groupby(['date','bins_size_riad']).RCB705.sum()

df_count = df_subset.groupby(['date','bins_size_riad']).RCB705.count()
df_count = np.cumsum(df_count.unstack(level = 0).fillna(0)) / df_count.unstack(level = 0).fillna(0).sum()

ind = np.arange(df_mean6.unstack(level = 0).iloc[:,0].shape[0])
width = 1/3
colors = ['white','grey','black']
hatches = ['\\','o','-']
lines = ['-','-.',':']

### Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitization of Residential Loans (By Bank-Size; Income Approach)')
ax.set_ylabel('Amount of securitization (in $1000; bars)')
ax.set(xlabel='Bank size (in mln $)')
for i in range(3):
    ax.bar([j + width * i for j in ind], df_mean6.unstack(level = 0).iloc[:,i], width, label = '{}'.format(2016 + i), \
           log = True, color = colors[i], edgecolor = 'black', hatch = hatches[i])
ax2 = ax.twinx()
ax2.set_ylabel('Percentage of banks; line')
for k in range(3):
    ax2.plot(ind, df_count.iloc[:,k], color = 'black', linestyle = lines[k])
ax2.axhline(0.5, alpha = 0.5, linestyle = '--', color = 'black')
ax2.text(x = ind[-1], y = 0.51, s = '50%', alpha = 0.5)
ax2.axhline(0.95, alpha = 0.5, linestyle = ':', color = 'black')
ax2.text(x = ind[-1], y = 0.96, s = '95%', alpha = 0.5)
ax.set_xticks([j + width * 1 for j in ind])
ax.set_xticklabels(np.round(np.exp(bins) / 1000, 2), rotation='vertical') # in millions
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=2)
plt.tight_layout()
plt.show()

fig.savefig('sec_banksize_riad.png')

## Figure (bar) loan sales (B705/B711) and size (Income approach)
### Prelims
df['lnTA_RIAD'] = np.log(df.TA_RIAD)
df = df[~(df.lnTA_RIAD == np.infty) & ~(df.lnTA_RIAD == -np.infty)]
df_subset = df[(df.date == 2016) | (df.date == 2017) | (df.date == 2018)]
df_subset['bins_size_riad'], bins = pd.cut(df_subset.lnTA_RIAD, 25, retbins = True)
df_mean6 = df_subset.groupby(['date','bins_size_riad']).RCB790.sum()

df_count = df_subset.groupby(['date','bins_size_riad']).RCB790.count()
df_count = np.cumsum(df_count.unstack(level = 0).fillna(0)) / df_count.unstack(level = 0).fillna(0).sum()

ind = np.arange(df_mean6.unstack(level = 0).iloc[:,0].shape[0])
width = 1/3
colors = ['white','grey','black']
hatches = ['\\','o','-']
lines = ['-','-.',':']

### Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Loans Sales with Recourse (By Bank-Size; Income approach)')
ax.set_ylabel('Amount of securitization (in $1000; bars)')
ax.set(xlabel='Bank size (in mln $)')
for i in range(3):
    ax.bar([j + width * i for j in ind], df_mean6.unstack(level = 0).iloc[:,i], width, label = '{}'.format(2016 + i), \
           log = True, color = colors[i], edgecolor = 'black', hatch = hatches[i])
ax2 = ax.twinx()
ax2.set_ylabel('Percentage of banks; line')
for k in range(3):
    ax2.plot(ind, df_count.iloc[:,k], color = 'black', linestyle = lines[k])
ax2.axhline(0.5, alpha = 0.5, linestyle = '--', color = 'black')
ax2.text(x = ind[-1], y = 0.51, s = '50%', alpha = 0.5)
ax2.axhline(0.95, alpha = 0.5, linestyle = ':', color = 'black')
ax2.text(x = ind[-1], y = 0.96, s = '95%', alpha = 0.5)
ax.set_xticks([j + width * 1 for j in ind])
ax.set_xticklabels(np.round(np.exp(bins) / 1000, 2), rotation='vertical') # in millions
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=2)
plt.tight_layout()
plt.show()

fig.savefig('ls_banksize_riad.png')