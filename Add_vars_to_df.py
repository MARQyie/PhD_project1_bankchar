#------------------------------------------
# Add new variables dataset US Call reports
# Mark van der Plaat
# Sep 2019 

''' 
    This document partially cleans the the dataset and add variables 
    for further analyses for working paper #1

    Data used: US Call reports 2001-2018 year-end
    Only insured, commercial banks are taken
    ''' 
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style='white',font_scale=1.5)

import os
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

#import sys # to use the help functions needed
#sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

#------------------------------------------
# Load df
df = pd.read_csv('Data\df_assetcomp_raw.csv', index_col = 0 )

#------------------------------------------
# Drop useless colums 
## Make list of variables to drop
''' The lists below make variable names that have been used to make new variables
    and are otherwise not needed.'''
var_codes = ['2170','3545','3548','0081','0071','2948','1771','1773','0213','1287','1754','1420',\
            '1460','1590','1797','2122','8725','8726','8727','8728','A126','A127','8723','8724',\
            'C968','C969','C970','C971','C972','C973','C974','C975','G641','B704','A222','3128',\
            'A223','B705','B706','B707','B708','B709','B710','B711','B790','B796','3210',\
            'B712','B713','B714','B715','B716','B717','B718',\
           'B719','B720','B721','B722','B723','B724','B725',\
           'B797','B798','B799','B800','B801','B802','B803',\
           'C393','C394','C395','C396','C397','C398','C399',\
           'C400','C401','C402','C403','C404','C405','C406',\
           'HU09','HU15'] 
scheme_codes = ['RCFD','RCON']
var_codes_rcr = ['7204','7205','7206']
scheme_codes_rcr = ['RCFA','RCFD','RCOA','RCON','RCO','RCF']

col_crop_rest = [j + i for i in var_codes for j in scheme_codes]
col_crop_rc = [j + i for i in var_codes_rcr for j in scheme_codes_rcr]
col_crop = col_crop_rest + col_crop_rc + ['RCFN2200','RCON2200','RCFAA223','RCOAA223','RCA223',\
                                          'RCFD1410','RC1771']

## Drop columns
df.drop(col_crop, axis = 1, inplace = True)

#------------------------------------------
# Check RIAD 4230 (are two variables with that name)
sum(df.RIAD4230_y.eq(df.RIAD4230_x) * 1) # two entries are not identical..

## Check the different entries
df.loc[~df.RIAD4230_y.eq(df.RIAD4230_x), ['RIAD4230_x','RIAD4230_y']]
'''There are some differences, but they are minor. Keep RIAD4320_y and rename'''
 
# Drop and rename
df.drop('RIAD4230_x', axis = 1, inplace = True)
df.rename(columns = {'RIAD4230_y':'RIAD4230'}, inplace = True)

#------------------------------------------
# Drop double RIAD4605
df.drop('RIAD4605_x', axis = 1, inplace = True)
df.rename(columns = {'RIAD4605_y':'RIAD4605'}, inplace = True)
#------------------------------------------
# Data selection
'''Only select insured, commercial banks with a physical location in the 50 states
    of the US.'''

## Select insured and commercial banks
'''RSSD9048 == 200, RSSD9424 != 0'''
df = df[(df.RSSD9048 == 200) & (df.RSSD9424 != 0)]

## Only take US states, no territories
''' Based on the state codes provided by the FRB'''
states = [i for i in range(1,57)]
df = df[df.RSSD9210.isin(states)]

#------------------------------------------
# Drop all banks with only three years of observations
drop_banks = np.array(df.IDRSSD.value_counts()[df.IDRSSD.value_counts() <= 3].index)
df = df[~df.IDRSSD.isin(drop_banks)]

#------------------------------------------
# Drop missings in Total assets, liquidity, loans, deposits, capital and income, RWA
## Sum cash and securities, label as liquid assets
df['liqass'] = df[['RC0071','RC0081','RC1773','RC1754']].sum(axis = 1, skipna = True) 
#RCON1350 (splits in BB987 + B989)

## Drop the NaNs
nandrop_cols = ['RC2170','RC2122','RC2200','RC3210','RIAD4340','RCG641','liqass']
df.dropna(subset = nandrop_cols , inplace = True)

#------------------------------------------
# Drop negative values in Total assets and loans
df = df[(df.RC2170 >= 0) | (df.RC2122 >= 0)]

#------------------------------------------
# Make new variables
## Loan Sales to TA
df['ls_sec_tot_ta'] = (df.ls_sec_tot / df.RC2170).replace(np.inf, 0)
df['ls_nonsec_tot_ta'] = (df.ls_nonsec_tot / df.RC2170).replace(np.inf, 0)
df['ls_tot_ta'] = (df.ls_tot / df.RC2170).replace(np.inf, 0)

## Bank size
df['size'] = np.log(df.RC2170)

## BHC indicator
df['bhc'] = (round(df.RSSD9364) > 0.0) * 1

## Inverse of TA
df['ta_inv'] = (1 / df.RC2170).replace(np.inf, 0)

## Bank size off- and on-balance
'''From Boyd & Gertler (1994)'''
df['tot_size'] = df.apply(lambda x: np.log(x.RC2170 * (1 + ((x.RIAD4079 - x.RIAD4080)/(x.RIAD4074 - x.RIAD4230))))\
  if ((x.RIAD4074 - x.RIAD4230) != 0) and (x.RC2170 * (1 + ((x.RIAD4079 - x.RIAD4080)/(x.RIAD4074 - x.RIAD4230)))) > 0\
  else 0.0 if ((x.RIAD4074 - x.RIAD4230) != 0) and ((x.RC2170 * (1 + ((x.RIAD4079 - x.RIAD4080)/(x.RIAD4074 - x.RIAD4230)))) == 0) else np.nan, axis = 1)

## Liquidity ratio
df['liqratio'] = (df.liqass / df.RC2170).replace(np.inf, 0)

## Trading assets ratio
df['traderatio'] = (df.RC3545 / df.RC2170).replace(np.inf, 0)

## Loan ratio
df['loanratio'] = (df.RC2122 / df.RC2170).replace(np.inf, 0)

## Loans-to-deposits
df['ltdratio'] = (df.RC2122 / df.RC2200).replace(np.inf, 0)

## Deposit ratio
df['depratio'] = (df.RC2200 / df.RC2170).replace(np.inf, 0)

## Share retail deposits
df['retaildep'] = ((df.RCONB549 + df.RCONB550) / df.RC2200).replace(np.inf, 0)

## Wholesale funding
'''First combine some variables'''
df['tot_time_dep'] = df[['RCON6648','RCON2604','RCON6648','RCONJ473','RCONJ474']].sum(axis = 1)
df['fed_repo'] = df[['RC2800','RCONB993','RCONB995']].sum(axis = 1)
df['wholesale'] = df[['tot_time_dep','fed_repo', 'RCON3200']].sum(axis = 1).\
                  divide(df.RC2170).replace(np.inf, 0)
                  
df['tot_time_dep_ta'] = df[['RCON6648','RCON2604','RCON6648','RCONJ473','RCONJ474']].sum(axis = 1).divide(df.RC2170).replace(np.inf, 0)
df['fed_repo_ta'] = df[['RC2800','RCONB993','RCONB995']].sum(axis = 1).divide(df.RC2170).replace(np.inf, 0)

## Simple equity ratio
df['eqratio'] = (df.RC3210 / df.RC2170).replace(np.inf, 0)

## Credit derivatives to total assets
df['cd_pur_ta'] = (df.cd_pur / df.RC2170).replace(np.inf, 0)
df['cd_sold_ta'] = (df.cd_sold / df.RC2170).replace(np.inf, 0)

## LN credit derivatives
df['ln_cd_pur'] = df.cd_pur.apply(lambda x: np.log(x) if x != 0.0 else 0.0)
df['ln_cd_sold'] = df.cd_sold.apply(lambda x: np.log(x) if x != 0.0 else 0.0)

## Loan growth
df['dloan'] = df.groupby('IDRSSD').RC2122.pct_change()

## Asset growth
df['dass'] = df.groupby('IDRSSD').RC2170.pct_change()

## Mortgages to loans
df['mortratio'] = (df[['RCON1415','RCONF158','RCONF159','RC1420','RC1460','RC1797']].sum(axis = 1).divide(df.RC2122)).replace(np.inf, 0)

## Home Equity Lines to loans
df['HEL'] = ((df.RCON3814) / df.RC2122).replace(np.inf, 0)

## Commercial loan ratio
df['comloanratio'] = (df.RCON1766 / df.RC2170).replace(np.inf, 0)

## Agricultural loan ratio
df['agriloanratio'] = (df.RC1590 / df.RC2170).replace(np.inf, 0)

## RWA over TA
df['rwata'] = (df.RCG641 / df.RC2170).replace(np.inf, 0)

## Loan charge-off ratio 
df['coffratio'] = (df.RIAD4635 / df.RC2122).replace(np.inf, 0)
df['coffratio_tot'] = (df[['RIADB747','RIADB748','RIADB749','RIADB750',\
  'RIADB751','RIADB752','RIADB753', 'RIAD4635']].sum(axis = 1) / df.RC2122).replace(np.inf, 0)

df['net_coffratio'] = ((df.RIAD4635 + df.RIAD4605)/ df.RC2122).replace(np.inf, 0)
df['net_coffratio_tot'] = ((df[['RIADB747','RIADB748','RIADB749','RIADB750',\
  'RIADB751','RIADB752','RIADB753', 'RIAD4635']].sum(axis = 1) - \
df[['RIADB754','RIADB755','RIADB756','RIADB757','RIADB758','RIADB759','RIADB760', 'RIAD4605']].\
sum(axis = 1))/ df.RC2122).replace(np.inf, 0)

df['net_coffratio_tot_ta'] = ((df[['RIADB747','RIADB748','RIADB749','RIADB750',\
  'RIADB751','RIADB752','RIADB753', 'RIAD4635']].sum(axis = 1) - \
df[['RIADB754','RIADB755','RIADB756','RIADB757','RIADB758','RIADB759','RIADB760', 'RIAD4605']].\
sum(axis = 1))/ df.RC2170).replace(np.inf, 0)

## Loan allowance ratio
df['allowratio_on_on'] = (df.RIAD3123 / df.RC2122).replace(np.inf, 0)
df['allowratio_off_on'] = (df.RCONB557/ df.RC2122).replace(np.inf, 0)
df['allowratio_tot_on'] = ((df.RIAD3123 + df.RCONB557) / df.RC2122).replace(np.inf, 0)

df['allowratio_on_off'] = (df.RIAD3123 / (df.ls_tot)).replace(np.inf, 0)
df['allowratio_off_off'] = (df.RCONB557/ (df.ls_tot)).replace(np.inf, 0)
df['allowratio_tot_off'] = ((df.RIAD3123 + df.RCONB557) / (df.ls_tot)).replace(np.inf, 0)

df['allowratio_on_tot'] = (df.RIAD3123 / (df.RC2122 + df.ls_tot)).replace(np.inf, 0)
df['allowratio_off_tot'] = (df.RCONB557/ (df.RC2122 + df.ls_tot)).replace(np.inf, 0)
df['allowratio_tot_tot'] = ((df.RIAD3123 + df.RCONB557) / (df.RC2122 + df.ls_tot)).replace(np.inf, 0)

df['tot_allowance'] = (df.RIAD3123 + df.RCONB557).replace(np.inf, 0)
df['allowratio_tot_ta'] = ((df.RIAD3123 + df.RCONB557) / df.RC2170).replace(np.inf, 0)

## Credit exposure loan sales ratio
df['lsseccredex_ratio'] = (df.ls_sec_credex / df.ls_sec_tot).replace(np.inf, 0)
df['lsnonseccredex_ratio'] = (df.ls_nonsec_credex / df.ls_nonsec_tot).replace(np.inf, 0)
df['lscredex_ratio'] = (df.ls_credex / df.ls_tot).replace(np.inf, 0)

## Loan provision ratio
df['provratio'] = (df.RIAD4230 / df.RC2122).replace(np.inf, 0)

## Interest expense relative to total liablities
df['intliabratio'] = (df.RIAD4079 / df.RC2948).replace(np.inf, 0)

## ROE
df['roe'] = (df.RIAD4340 / df.RC3210).replace(np.inf, 0)

## ROA 
df['roa'] = (df.RIAD4340 / df.RC2170).replace(np.inf, 0)

## Net interest margin
df['nim'] = (df.RIAD4074 / df.RC2170).replace(np.inf, 0)

## Cost-to-income margin
df['costinc'] = (df.RIAD4093 / (df.RIAD4074 + df.RIAD4079)).replace(np.inf, 0)

## Non-interest income to operating income
df['nonincoperinc'] = (df.RIAD4079 / (df.RIAD4074 + df.RIAD4079)).replace(np.inf, 0)

## Loan interest income
df['intincloan'] = ((df.RIAD4010 + df.RIAD4065) / df.RIAD4107).replace(np.inf, 0)

## Interest income of other deposit institutions
df['intincdepins'] = (df.RIAD4115 / df.RIAD4107).replace(np.inf, 0)

## Interest income of securities
df['intincsec'] = ((df.RIADB488 + df.RIADB489 + df.RIAD4060) / df.RIAD4107).replace(np.inf, 0)

## Interest income of trading assets
df['intinctrade'] = (df.RIAD4069 / df.RIAD4107).replace(np.inf, 0)

## Interest income of REPO
df['intincrepo'] = (df.RIAD4020 / df.RIAD4107).replace(np.inf, 0)

## Other interest income
df['intincoth'] = (df.RIAD4518 / df.RIAD4107).replace(np.inf, 0)

## Interest expenses on REPOs
df['intexprepo'] = (df.RIAD4180 / df.RIAD4073).replace(np.inf, 0)

## Interest expenses on trading liabilities
df['intexptrade'] = (df.RIAD4185 / df.RIAD4073).replace(np.inf, 0)

## Interest expenses on subordinated notes
df['intexpsub'] = (df.RIAD4200 / df.RIAD4073).replace(np.inf, 0)

#-------------------------------------------------
# Add Summary of Deposit data
from Code_docs.help_functions.Proxies_org_complex_banks import LimitedServiceBranches,\
     spatialComplexity, noBranchBank, readSODFiles, maxDistanceBranches

## Number of limited service, full service and total branches
df_branches = LimitedServiceBranches(2001,2019)
df_branches.reset_index(inplace = True)

## Number of States Active
df_complex = spatialComplexity(2001,2019)
df_complex = df_complex.reset_index()

## No Bank Branches
df_nobranch = noBranchBank(2001,2019)
df_nobranch = df_nobranch.reset_index()

## Mean Distance Branch, Head quarters
df_distance = maxDistanceBranches()
df_distance = df_distance.reset_index()
df_distance.rename(columns = {0:'distance'}, inplace = True)

## Add the dfs to the dataframe
df = df.merge(df_branches, on = ['IDRSSD', 'date'], how = 'left')
df = df.merge(df_complex, on = ['IDRSSD', 'date'], how = 'left')
df = df.merge(df_nobranch, on = ['IDRSSD', 'date'], how = 'left')
df = df.merge(df_distance, on = ['IDRSSD', 'date'], how = 'left')

## Drop banks that have no known number of branches
df.dropna(subset = ['num_branch'], inplace = True)

#------------------------------------------
# Change the name column name
df.rename(columns = {'Financial Institution Name':'name'}, inplace = True)

#------------------------------------------
## Save df
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
df.to_csv('Data\df_wp1_newvars.csv')

