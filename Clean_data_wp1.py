#------------------------------------------
# Clean dataset US Call reports
# Mark van der Plaat
# Sep 2019 

''' 
    This document cleans the the dataset for further analyses for working paper #1

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
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

import sys # to use the help functions needed
sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

#------------------------------------------
# Import help functions
from Number_of_branches import numberOfBranches

#------------------------------------------
# Load df
df = pd.read_csv('df_assetcomp_raw.csv', index_col = 0 )

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
# Drop all banks with only one year of observations
drop_banks = np.array(df.IDRSSD.value_counts()[df.IDRSSD.value_counts() == 1].index)
df = df[~df.IDRSSD.isin(drop_banks)]

#------------------------------------------
# Drop missings in Total assets, liquidity, loans, deposits, capital and income, RWA
## Sum cash and securities, label as liquid assets
df['liqass'] = df[['RC0071','RC0081','RC1773','RC1754']].sum(axis = 1, skipna = True) 
#TODO add RCON1350 (splits in BB987 + B989)

## Drop the NaNs
nandrop_cols = ['RC2170','RC2122','RC2200','RC3210','RIAD4340','RCG641','liqass']
df.dropna(subset = nandrop_cols , inplace = True)

#------------------------------------------
# Make new variables
## Bank size
df['size'] = np.log(df.RC2170)

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
df['tot_time_dep'] = df.apply(lambda x: np.sum(x[['RCON6648','RCON2604']]) if x.date < 2010  \
                     else np.sum(x[['RCON6648','RCONJ473','RCONJ474']]), axis = 1)
df['fed_repo'] = df.apply(lambda x: x.RC2800 if x.date == 2001 else np.sum(x[['RCONB993','RCONB995']]), axis = 1)

df['wholesale'] = df[['tot_time_dep','fed_repo', 'RCON3200']].aggregate(np.sum).\
                  divide(df.RC2170).replace(np.inf, 0)

## Simple equity ratio
df['eqratio'] = (df.RC3210 / df.RC2170).replace(np.inf, 0)

## Loan growth
df['dloan'] = df.groupby('IDRSSD').RC2122.pct_change()

## Asset growth
df['dass'] = df.groupby('IDRSSD').RC2170.pct_change()

## Mortgages to loans
df['mortratio'] = df.apply(lambda x: np.sum(x[['RC1415','RC1420','RC1460','RC1797']]) / x.RC2122 \
                  if x.date < 2008 else np.sum(x[['RCONF158','RCONF149','RC1420','RC1460','RC1797']])\
                  / x.RC2122, axis = 1).replace(np.inf, 0)

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

## Credit exposure loan sales ratio
df['lsseccredex_ratio'] = (df.ls_sec_credex / df.ls_tot).replace(np.inf, 0)
df['lsnonseccredex_ratio'] = (df.ls_nonsec_credex / df.ls_tot).replace(np.inf, 0)
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

## Number of branches
df_branches = numberOfBranches(2001,2019)
df_branches.reset_index(inplace = True)

df = df.merge(df_branches, on = ['IDRSSD', 'date'], how = 'left')

### Drop banks that have no known number of branches
df.dropna(subset = ['num_branch'], inplace = True)

#------------------------------------------
# Limit balance sheet ratios ratios between 0,1 interval
vars_limit = ['liqratio','traderatio','loanratio','retaildep','eqratio','comloanratio',\
              'agriloanratio']

for i in vars_limit:
    df = df[df['{}'.format(i)].between(0,1,inclusive = True)]

#------------------------------------------
# Take out the infinite values
df = df.replace([np.inf, -np.inf], np.nan)
    
#------------------------------------------
# Check outliers
# Residential non-securitized loan sales
sns.boxplot(df.RCB790) # one huge one, take out
df = df[df.RCB790 != df.RCB790.max()]
 
## Loan-to-deposit ratio
sns.boxplot(df.ltdratio) # quite some
df = df[df.ltdratio < df.ltdratio.quantile(q = 0.999)]

## RWATA
sns.boxplot(df.rwata) # There are a handful outliers. No action yet

## Loan charge-off ratio 
sns.boxplot(df.coffratio) # One mega outlier, take out
df = df[df.coffratio != df.coffratio.max()]

## Loan allowance ratio 
# TODO
#sns.boxplot(df.allowratio) # No weird obs, no action

## Loan provision ratio
sns.boxplot(df.provratio) # Some outliers, 4 impossibly negative, take out
df = df[~df.provratio.isin(df.provratio.nsmallest(n = 4))]

## ROE
'''This variable looks very disperse with many highly negative values.
    Might not be suitable for an analysis.'''
sns.boxplot(df.roe) # Three huge negative values, take out
#df = df[~df.roe.isin(df.roe.nsmallest(n = 3))]

## ROA 
sns.boxplot(df.roa) # No weird obs

## Net interest margin
sns.boxplot(df.nim) # No weird obs

## Cost-to-income margin
sns.boxplot(df.costinc) # Looks non-standard, no action

## Non-interest income to operating income
sns.boxplot(df.nonincoperinc) # Looks non-standard, no action

## Loan interest income
sns.boxplot(df.intincloan) # No action

## Interest income of other deposit institutions
sns.boxplot(df.intincdepins) # No action

## Interest income of securities
sns.boxplot(df.intincsec) # No action

## Interest income of trading assets
sns.boxplot(df.intinctrade) # No action

## Interest income of REPO
sns.boxplot(df.intincrepo) # No action

## Other interest income
sns.boxplot(df.intincoth) # No action

## Interest expenses on REPOs
sns.boxplot(df.intexprepo) # No action

## Interest expenses on trading liabilities
sns.boxplot(df.intexptrade) # Two huge outliers, take out
df = df[df.intexptrade != df.intexptrade.max()]
df = df[df.intexptrade != df.intexptrade.min()]

## Interest expenses on subordinated notes
sns.boxplot(df.intexpsub) # No action

## Number of branches
sns.boxplot(df.num_branch) # No action

## Regulatory Capital Ratios
'''Kicks out many obervations with values at sec_tot'''
sns.boxplot(df.RC7205) # many large outliers
df = df[df.RC7205 < df.RC7205.quantile(q = 0.999)]

sns.boxplot(df.RC7206) # No action

#------------------------------------------
# Check obs per years
year_obs = df.date.value_counts()

#------------------------------------------
## Save df
df.to_csv('df_wp1_clean.csv')

