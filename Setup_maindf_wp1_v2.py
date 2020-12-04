#------------------------------------------
# Setup the DataFrame for wp1
# Mark van der Plaat
# Sep 2019; data update: Mar 2020 

''' 
    This document partially cleans the the dataset and add variables 
    for further analyses for working paper #1

    Data used: US Call reports 2001-2018 year-end
    Only insured, commercial banks are taken
    ''' 
#------------------------------------------------------------
# Import Packages
#------------------------------------------------------------
    
import pandas as pd
import numpy as np

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

#import sys # to use the help functions needed
#sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

#------------------------------------------------------------
# Load Data
#------------------------------------------------------------
df_raw = pd.read_csv('Data\df_wp1_raw.csv')

#------------------------------------------------------------
# Filter Data
#------------------------------------------------------------
'''Only select insured, commercial banks with a physical location in the 50 states
    of the US.'''

## Select insured and commercial banks
'''RSSD9048 == 200, RSSD9424 != 0'''
df_filtered = df_raw[(df_raw.RSSD9048 == 200) & (df_raw.RSSD9424 != 0)]

## Only take US states, no territories
''' Based on the state codes provided by the FRB'''
df_filtered = df_filtered[df_filtered.RSSD9210.isin(range(1,57))]

#------------------------------------------
# Drop all banks with three years or fewer of observations
drop_banks = df_filtered.IDRSSD.value_counts()[df_filtered.IDRSSD.value_counts() <= 3].index.unique().tolist()
df_filtered = df_filtered[~df_filtered.IDRSSD.isin(drop_banks)]

#------------------------------------------
# Drop missings in Total assets, loans, deposits, income
nandrop_cols = ['RC2170','RC2122','RC2200','RIAD4340']
df_filtered.dropna(subset = nandrop_cols , inplace = True)

#------------------------------------------
# Drop negative and zero values in Total assets and loans
df_filtered = df_filtered[(df_filtered.RC2170 > 0) & (df_filtered.RC2122 > 0)]

#------------------------------------------------------------
# Create new df and variabless
#------------------------------------------------------------

#------------------------------------------
# Create new df

df = df_filtered[['IDRSSD','date']]

#------------------------------------------
# Make new variables

# Dependent Variables
## Total Charge-offs
df.loc[:,'net_coff_tot'] = (df_filtered[['RIADB747','RIADB748','RIADB749','RIADB750',\
  'RIADB751','RIADB752','RIADB753', 'RIAD4635']].sum(axis = 1) - \
df_filtered[['RIADB754','RIADB755','RIADB756','RIADB757','RIADB758','RIADB759','RIADB760', 'RIAD4605']].\
sum(axis = 1)) / df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
                                    'RCB711','RCONFT08','RCONFT10','RC2122','RC2123']].sum(axis = 1, skipna = True)
                                    
### With and without loan transfer exposure
df.loc[:,'net_coff_on'] = (df_filtered.RIAD4635 - df_filtered.RIAD4605) / df_filtered[['RC2122', 'RC2123']].sum(axis = 1)

df.loc[:,'net_coff_off'] = ((df_filtered[['RIADB747','RIADB748','RIADB749','RIADB750',\
  'RIADB751','RIADB752','RIADB753']].sum(axis = 1) - \
df_filtered[['RIADB754','RIADB755','RIADB756','RIADB757','RIADB758','RIADB759','RIADB760']].\
sum(axis = 1)) / df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
                                    'RCB711','RCONFT08','RCONFT10']].sum(axis = 1, skipna = True)).replace([np.inf,-np.inf, np.nan], 0)
                                    

## Non-performing loans
## NOTE: We classify NPL as 90 and still accruing + nonaccrual. Just to be sure we also calculate 30-89 days accruing loans
df.loc[:,'npl30'] = df_filtered.loc[:,['RCON2759','RCON3493','RCON5398','RCON5401','RCON3499','RCON3502',\
              'RCFD5377','RC5380','RC1594','RCFD1251','RC1254','RCB575','RCB578',\
              'RC5389','RC5459','RCFD1257','RC1271','RCONF172','RCONF173','RCONC236',\
              'RCONC238','RCONF178','RCONF179','RCK213','RCK216']].sum(axis = 1, skipna = True)
df.loc[:,'npl90'] = df_filtered.loc[:,['RCON2769','RCON3494','RCON5399','RCON5402','RCON3500','RCON3503',\
              'RCFD5378','RC5381','RC1597','RCFD1252','RC1255','RCB576','RCB579',\
              'RC5390','RC5460','RCFD1258','RC1272','RCONF174','RCONF175','RCONC237',\
              'RCONC239','RCONF180','RCONF181','RCK214','RCK217']].sum(axis = 1, skipna = True)
df.loc[:,'nplna'] = df_filtered.loc[:,['RCON3492','RCON3495','RCON5400','RCON5403','RCON3501','RCON3504',\
              'RCFD5379','RC5382','RC1583','RCFD1253','RC1256','RCB577','RCB580',\
              'RC5391','RC5461','RCFD1259','RC1791','RCONF176','RCONF177','RCONC229',\
              'RCONC230','RCONF182','RCONF183','RCK215','RCK218']].sum(axis = 1, skipna = True)
df.loc[:,'npl_on'] = (df.npl90 + df.nplna) / df_filtered[['RC2122', 'RC2123']].sum(axis = 1)

## OBS and tot non-performing loans
df.loc[:,'npl_off'] = (df_filtered.loc[:,['RCB740','RCB741','RCB742','RCB743','RCB744','RCB745','RCB746']].sum(axis = 1, skipna = True).divide(df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711','RCONFT08']].sum(axis = 1, skipna = True))).replace([np.inf,-np.inf, np.nan], 0)

df.loc[:,'npl_tot'] = (df.npl90 + df.nplna + df_filtered.loc[:,['RCB740','RCB741','RCB742','RCB743','RCB744','RCB745','RCB746']].sum(axis = 1, skipna = True)).divide(df_filtered.loc[:,['RC2122', 'RC2123','RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711','RCONFT08']].sum(axis = 1, skipna = True))

## Deliquent and Defaulted OBS loans
## NOTE: OBS npl + charge-offs
df.loc[:,'ddl_off'] = (df_filtered.loc[:,['RCB740','RCB741','RCB742','RCB743',\
                      'RCB744','RCB745','RCB746']].sum(axis = 1, skipna = True) +\
                       df_filtered[['RIADB747','RIADB748','RIADB749','RIADB750',\
                      'RIADB751','RIADB752','RIADB753']].sum(axis = 1) - \
                       df_filtered[['RIADB754','RIADB755','RIADB756','RIADB757',\
                                    'RIADB758','RIADB759','RIADB760']].sum(axis = 1)).\
                       divide(df_filtered.loc[:,['RCB705','RCB706',\
                      'RCB707','RCB708','RCB709','RCB710','RCB711','RCONFT08']].\
                       sum(axis = 1, skipna = True)).fillna(0.0).replace([np.inf, -np.inf], 0)

## RWATA
df.loc[:,'rwata'] = df_filtered.RCG641 / df_filtered.RC2170

## Allowance Ratio (On balance)
df.loc[:,'allow_on'] = df_filtered.RIAD3123.divide(df_filtered[['RC2122', 'RC2123']].sum(axis = 1))

## Allowance Ratio (OBS)
### Total assets
df.loc[:,'allow_off_ta'] = df_filtered.RCB557.divide(df_filtered.RC2170).replace([np.inf,-np.inf, np.nan], 0)

### Revenue based 
### From Boyd & Gertler (1994). Used by: Clark & Siems (2002), Calmès & Théoret (2010)
df.loc[:,'obs_ta_rb'] = df_filtered.RC2170.multiply(df_filtered.RIAD4079.divide(df_filtered.RIAD4074.subtract(df_filtered.RIAD4230))).replace([np.inf,-np.inf], 0).abs()
df.loc[:,'allow_off_rb'] = df_filtered.RCB557.divide(df.obs_ta_rb).replace([np.inf,-np.inf, np.nan], 0)

### OBS items
### Based on Papanikolaou & Wolff (2014)
df.loc[:,'obs_ta_items'] = df_filtered.loc[:,['RC3814', 'RC3815', 'RC3816', 'RC6550',\
                        'RC3817', 'RC3818', 'RC3819', 'RC3821', 'RC3411', 'RC3428', 'RC3433',\
                        'RCA534', 'RCA535', 'RC3430', 'RC5591', 'RCA126', 'RCA127', 'RC8723',\
                        'RC8724', 'RC8725', 'RC8726', 'RC8727', 'RC8728', 'RCF164', 'RCF165',\
                        'RCJ457', 'RCJ458', 'RCJ459', 'RCC968', 'RCC969',\
                        'RCC970', 'RCC971', 'RCC972', 'RCC973', 'RCC974', 'RCC975', 'RCB705',\
                        'RCB706', 'RCB707', 'RCB708', 'RCB709', 'RCB710', 'RCB711', 'RCONFT08',\
                        'RCB790', 'RCB791', 'RCB792', 'RCB793', 'RCB794', 'RCB795', 'RCB796',\
                        'RCONFT10']].sum(axis = 1)
                                    
df.loc[:,'allow_off_items'] = df_filtered.RCB557.divide(df.obs_ta_items).replace([np.inf,-np.inf, np.nan], 0)

### Credit Equivalent Amounts
### Based on Clark & Siems (2002)
df.loc[:,'obs_ta_cea'] = df_filtered.loc[:,['RCB645','RCB650','RCB655','RCB660','RCB664',\
                     'RCB676','RCB682','RCB687','RCA167','RCG592',\
                     'RCD992','RCD998','RCG607','RCS516','RCG619','RCS526',\
                     'RCG625', 'RCS541','RCS542','RCS549']].sum(axis = 1)

df.loc[:,'allow_off_cea'] = df_filtered.RCB557.divide(df.obs_ta_cea).replace([np.inf,-np.inf, np.nan], 0)

## Provision Ratio
df.loc[:,'prov_ratio'] = df_filtered.RIAD4230.divide(df_filtered[['RC2122', 'RC2123']].sum(axis = 1))

# Control variables Baseline
## Loan sales
df.loc[:,'ls_tot'] = df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
               'RCB711','RCB790','RCB791','RCB792','RCB793','RCB794','RCB795',\
                  'RCB796','RCONFT08','RCONFT10']].sum(axis = 1, skipna = True) / df_filtered.RC2170
          
## Regulatory Leverage Ratio and capital ratio
df.loc[:,'reg_lev'] = df_filtered.RC7204
df.loc[:,'reg_cap'] = df_filtered.RC7205

## Equity ratio
df.loc[:,'eqratio'] = df_filtered.RC3210.divide(df_filtered.RC2170)

## Loan level
df.loc[:,'loanlevel'] = df_filtered[['RC2122', 'RC2123']].sum(axis = 1)

## Loan Ratio
df.loc[:,'loanratio'] = df_filtered[['RC2122', 'RC2123']].sum(axis = 1) / df_filtered.RC2170

## Loan to deposits
df.loc[:,'loandep'] = df_filtered[['RC2122', 'RC2123']].sum(axis = 1) / df_filtered.RC2200

## ROA
df.loc[:,'roa'] = df_filtered.RIAD4340 / df_filtered.RC2170

## Deposit Ratio
df.loc[:,'depratio'] = df_filtered.RC2200 / df_filtered.RC2170

## Commercial Loan Ratio
df.loc[:,'comloanratio'] = df_filtered.loc[:,'RCON1766'] / df_filtered[['RC2122', 'RC2123']].sum(axis = 1)

## Mortgage Ratio
df_filtered.loc[:,'loans_sec_land'] = df_filtered.apply(lambda x: x.RCON1415 if ~np.isnan(x.RCON1415) else x[['RCF158','RCF159']].sum(), axis = 1) # Not needed in analysis
df_filtered.loc[:,'loans_sec_nonfarm'] = df_filtered.apply(lambda x: x.RCON1480 if ~np.isnan(x.RCON1480) else x[['RCF160','RCF161']].sum(), axis = 1) # Not needed in analysis
df.loc[:,'mortratio'] = df_filtered.loc[:,['loans_sec_land','RC1420','RC1460','RC1797',\
  'RC5367','RC5368','loans_sec_nonfarm']].sum(axis = 1).divide(df_filtered[['RC2122', 'RC2123']].sum(axis = 1))

## Consumer Loan Ratio
df.loc[:,'consloanratio'] = df_filtered.loc[:,['RCB538','RCB539','RC2011','RCK137','RCK207']].sum(axis = 1).divide(df_filtered[['RC2122', 'RC2123']].sum(axis = 1))

## Loans HHI
df.loc[:,'agriloanratio'] = df_filtered.RC1590 / df_filtered[['RC2122', 'RC2123']].sum(axis = 1) # Not needed in analysis, but needed to calculate loan HHI
df.loc[:,'loanhhi'] = df.loc[:,['mortratio','consloanratio','comloanratio','agriloanratio']].pow(2).sum(axis = 1) 

## Cost-to-income (costs / operating income = costs / (net int. inc + non-int inc))
df.loc[:,'costinc'] = ((df_filtered.RIAD4093) / (df_filtered.RIAD4074 + df_filtered.RIAD4079)).replace(np.inf, np.nan)

## Size
df.loc[:,'size'] = df_filtered.RC2170 

## BHC Indicator
df.loc[:,'bhc'] = (round(df_filtered.RSSD9364) > 0.0) * 1

# Variables for Robustness
## Securitization
df.loc[:,'ls_sec'] = df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
               'RCB711','RCONFT08']].sum(axis = 1, skipna = True)\
                / df_filtered.RC2170
df.loc[:,'ls_nonsec'] = df_filtered.loc[:,['RCB790','RCB791','RCB792','RCB793','RCB794','RCB795',\
                  'RCB796','RCONFT10']].sum(axis = 1, skipna = True) / df_filtered.RC2170
  
## Dummy Loan sales
df.loc[:,'ls_dum'] = (df.ls_tot > 0.0) * 1
  
## Credit Exposure
### Calculate underlying exposures (interest-only strips, sub. notes/standby letters of credit after 2017 not available)
df.loc[:,'mace_sec_io'] = df_filtered.loc[:,['RCB712','RCB713','RCB714','RCB715','RCB716','RCB717','RCB718']].sum(axis = 1, skipna = True)
df.loc[:,'mace_sec_subloc'] = df_filtered.loc[:,['RCB719','RCB720','RCB721','RCB722','RCB723','RCB724','RCB725',\
                                                 'RCC393','RCC394','RCC395','RCC396','RCC397','RCC398','RCC399',\
                                                 'RCC400','RCC401','RCC402','RCC403','RCC404','RCC405','RCC406']].sum(axis = 1, skipna = True)
df.loc[:,'mace_sec'] = df_filtered.loc[:,['RCB712','RCB713','RCB714','RCB715','RCB716','RCB717','RCB718',\
                                          'RCB719','RCB720','RCB721','RCB722','RCB723','RCB724','RCB725',\
                                          'RCC393','RCC394','RCC395','RCC396','RCC397','RCC398','RCC399',\
                                          'RCC400','RCC401','RCC402','RCC403','RCC404','RCC405','RCC406',\
                                          'RCHU09','RCFDHU10','RCFDHU11','RCFDHU12','RCFDHU13','RCFDHU14','RCHU15']].sum(axis = 1, skipna = True)
df.loc[:,'mace_nonsec'] = df_filtered.loc[:,['RCB797','RCB798','RCB799','RCB800','RCB801','RCB802','RCB803']].sum(axis = 1, skipna = True)

### Credit exposure
df.loc[:,'credex_sec'] = (df.loc[:,'mace_sec'] / df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
               'RCB711','RCONFT08']].sum(axis = 1, skipna = True)).replace(np.inf, 0).replace(np.nan, 0)
df['credex_nonsec'] = (df.loc[:,'mace_nonsec'] /  df_filtered.loc[:,['RCB790','RCB791','RCB792','RCB793','RCB794','RCB795',\
                  'RCB796','RCONFT10']].sum(axis = 1, skipna = True)).replace(np.inf, 0).replace(np.nan, 0)
df['credex_sbo'] = (df_filtered.loc[:,'RCA250'] / df_filtered.loc[:,'RCA249']).replace(np.nan, 0)
    
df['credex_tot'] = (df.loc[:,['mace_sec','mace_nonsec']].sum(axis = 1, skipna = True).add(df_filtered.loc[:,'RCA250']) / df_filtered.loc[:,['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710',\
               'RCB711','RCB790','RCB791','RCB792','RCB793','RCB794','RCB795',\
                  'RCB796','RCONFT08','RCONFT10','RCA249']].sum(axis = 1, skipna = True)).replace(np.inf, 0).replace(np.nan, 0)
df['credex_tot_alt'] = (df.loc[:,['mace_sec','mace_nonsec']].sum(axis = 1, skipna = True).add(df_filtered.loc[:,['RCA250','RCB745','RCB746','RCB776','RCB777','RCB778','RCB779','RCB780']].sum(axis = 1, skipna = True)) / df.obs_ta_cea).replace([np.inf,-np.inf, np.nan], 0)
    
'''NOTE: Any variables for figures could be taken from the filtered raw data 
    and the final data'''

#------------------------------------------------------------
# Add variables for measuring the business cycle.
#------------------------------------------------------------
# VIX data (Yahoo finance)
df_vix = pd.read_csv('D:\RUG\Data\Time_series\VIX\VIX.csv')

## Make Date datetime
df_vix.Date = pd.to_datetime(df_vix.Date)

## Transform to yearly data
### Take mean
df_vix_mean = df_vix.groupby(df_vix.Date.dt.year).mean()

### Take last
df_vix_last = df_vix.groupby(df_vix.Date.dt.year).last()

### Merge datasets
df.loc[:,'vix_mean'] = df_vix_mean.loc[df.date, 'Close'].values
df.loc[:,'vix_last'] = df_vix_last.loc[df.date, 'Close'].values

# Data Total Factor Productivity (FRB San Francisco)
df_tfp = pd.read_excel('D:\RUG\Data\Time_series\FRBSF\quarterly_tfp.xlsx', sheet_name = 'annual')
df_tfp.index = df_tfp.date

df.loc[:,'tfp'] = df_tfp.loc[df.date,'dtfp_util'].values

# FRED data sets
## Producer Confidence
df_fred_pc = pd.read_csv('D:\RUG\Data\Time_series\FRED\FRED_BTS_prod_confidence.csv')
df_fred_pc.DATE = pd.to_datetime(df_fred_pc.DATE)

### Take mean and last
df_fred_pc_mean = (df_fred_pc.groupby(df_fred_pc.DATE.dt.year).BSCICP03USM665S.mean() - 1e2) / 1e2
df_fred_pc_last = (df_fred_pc.groupby(df_fred_pc.DATE.dt.year).BSCICP03USM665S.last() - 1e2) / 1e2

### Merge datasets
df.loc[:,'pc_mean'] = df_fred_pc_mean.loc[df.date].values
df.loc[:,'pc_last'] = df_fred_pc_last.loc[df.date].values

## GDP Growth
df_fred_gdp = pd.read_csv('D:\RUG\Data\Time_series\World_Bank\API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_1740234.csv', skiprows = [0, 1, 2, 3])

### Get the US 
df_fred_gdp_growth = df_fred_gdp.loc[df_fred_gdp['Country Code'] == 'USA',list(map(str, range(1960,2019+1)))].T

### Change index to int
df_fred_gdp_growth.index = df_fred_gdp_growth.index.astype(int)

### Merge datasets
df.loc[:,'gdp'] = df_fred_gdp_growth.loc[df.date].values

## Consumer Sentiment
df_fred_cs = pd.read_csv('D:\RUG\Data\Time_series\FRED\FRED_UniMichigan_consumer_sentiment.csv', na_values ='.')
df_fred_cs.dropna(inplace = True)
df_fred_cs.DATE = pd.to_datetime(df_fred_cs.DATE)

### Take mean and last
df_fred_cs_mean = (df_fred_cs.groupby(df_fred_cs.DATE.dt.year).UMCSENT.mean() - 1e2) / 1e2
df_fred_cs_last = (df_fred_cs.groupby(df_fred_cs.DATE.dt.year).UMCSENT.last() - 1e2) / 1e2

### Merge datasets
df.loc[:,'cs_mean'] = df_fred_cs_mean.loc[df.date].values
df.loc[:,'cs_last'] = df_fred_cs_mean.loc[df.date].values    

## Real investment growth
df_fred_rig = pd.read_csv('D:\RUG\Data\Time_series\FRED\FRED_RGPDI.csv')
df_fred_rig.DATE = pd.to_datetime(df_fred_rig.DATE)

### Take mean
df_fred_rig_mean = df_fred_rig.groupby(df_fred_rig.DATE.dt.year).A006RL1A225NBEA.mean() / 100

### Merge datasets
df.loc[:,'rig_mean'] = df_fred_rig_mean.loc[df.date].values


#------------------------------------------------------------
# Add Dodd-Frank dummy, see Bordo & Duca (2018; NBER)
#------------------------------------------------------------

df.loc[:,'dodd'] = [1 if date > 2010 else .5 if date == 2010 else 0 for date in df.date]

''' 
# Variables for instruments
## Employees and log Employees
df.loc[:,'empl'] = df_filtered.RIAD4150
df.loc[:,'log_empl'] = np.log(df_filtered.RIAD4150 + 1)
   
from Code_docs.help_functions.Proxies_org_complex_banks import LimitedServiceBranches,\
     spatialComplexity, noBranchBank, readSODFiles

## Number of limited service, full service and total branches
df_branches = LimitedServiceBranches(2001,2020)
df_branches.reset_index(inplace = True)
df_branches.loc[:,'log_num_branch'] = np.log(df_branches.num_branch)

## Number of States Active
df_complex = spatialComplexity(2001,2020)
df_complex = df_complex.reset_index()
df_complex.loc[:,'log_states'] = np.log(df_complex.unique_states)

## No Bank Branches
df_nobranch = noBranchBank(2001,2020)
df_nobranch = df_nobranch.reset_index()

## Add the dfs to the dataframe
df = df.merge(df_branches, on = ['IDRSSD', 'date'], how = 'left')
df = df.merge(df_complex, on = ['IDRSSD', 'date'], how = 'left')
df = df.merge(df_nobranch, on = ['IDRSSD', 'date'], how = 'left')
'''
#------------------------------------------------------------
# Drop any nans in baseline variables
#------------------------------------------------------------
vars_baseline = ['net_coff_tot','npl_on','ls_tot','reg_lev','reg_cap','loanratio','roa','depratio',\
                 'comloanratio','mortratio','loanhhi','costinc','size','bhc']
df.dropna(subset = vars_baseline, inplace = True)  

#------------------------------------------------------------
# Remove outliers
#------------------------------------------------------------           

# Loan sales
df = df.drop(df.ls_tot.idxmax())

# Net Charge_offs
df = df.drop(df.net_coff_tot.idxmax())

'''
## Remove big outlier num_branch
df = df[df.num_branch != df.num_branch.max()] '''

## Limit ROA and prov ratio to [-1,1]
vars_limit = ['roa']

for i in vars_limit:
    df = df[df['{}'.format(i)].between(-1,1,inclusive = True)]
    
# Limit vars to [0,1] 
vars_limit = ['loanratio','reg_cap','mortratio','allow_off_cea']

for i in vars_limit:
    df = df[df['{}'.format(i)].between(0,1,inclusive = True)]   
    

#------------------------------------------
## Save df
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')
df_filtered.to_csv('Data\df_wp1_filtered.csv', index = False)
df.to_csv('Data\df_wp1_main.csv', index = False)

