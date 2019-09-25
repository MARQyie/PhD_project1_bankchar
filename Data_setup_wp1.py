#------------------------------------------
# Setup dataset US Call reports
# Mark van der Plaat
# August 2019 

''' 
    This document sets up the dataset for further analyses for working paper #1

    Data used: US Call reports 2001-2018 year-end
    Only insured, commercial banks are taken
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

import csv

#------------------------------------------
# Set Prelims 
start = 2001
end = 2019
path_info = r'X:/My Documents/Data/Data_call_reports_fed'
path_call = r'X:/My Documents/Data/Data_call_reports_FFIEC2'

#------------------------------------------
# Set filenames
file_info1 = r'/call0{}12.xpt'
file_info2 = r'/call{}12.xpt'
file_rc = r'/{}/FFIEC CDR Call Schedule RC 1231{}.txt'
file_rcb1 = r'/{}/FFIEC CDR Call Schedule RCB 1231{}.txt'
file_rcb2 = r'/{}/FFIEC CDR Call Schedule RCB 1231{}(1 of 2).txt'
file_rcc = r'/{}/FFIEC CDR Call Schedule RCCI 1231{}.txt'
file_rce = r'/{}/FFIEC CDR Call Schedule RCE 1231{}.txt'
file_rcg = r'/{}/FFIEC CDR Call Schedule RCG 1231{}.txt'
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
file_rcri1 = r'/{}/FFIEC CDR Call Schedule RCRIB 1231{}.txt' #2014
file_rcri2 = r'/{}/FFIEC CDR Call Schedule RCRI 1231{}.txt' #from 2015
file_rcs = r'/{}/FFIEC CDR Call Schedule RCS 1231{}.txt'
file_ri = r'/{}/FFIEC CDR Call Schedule RI 1231{}.txt'
file_ribi = r'/{}/FFIEC CDR Call Schedule RIBI 1231{}.txt'
file_ribii = r'/{}/FFIEC CDR Call Schedule RIBII 1231{}.txt'

#------------------------------------------
# Set variables needed per csv
## From balance sheet
vars_info = ['RSSD9001','RSSD9999','RSSD9048','RSSD9424','RSSD9170','RSSD9210']
vars_rc = '|'.join(['IDRSSD','2170','3545','3548','2200','0081','0071','3210','2948',\
                    '2800','B993','B995','3200']) 
vars_rcb = '|'.join(['IDRSSD','1771','1773','0213','1287','1754'])
vars_rcc = '|'.join(['IDRSSD','1410','1415','1420','1797','1460','1288','1590','1766',\
                     '2122','1590','F158','F159']) 
vars_rce = '|'.join(['IDRSSD','B549','B550','6648','2604','J473','J474','B535','2081'])
vars_rcg = '|'.join(['IDRSSD', 'B557'])
vars_rcl = '|'.join(['IDRSSD','8725','8726','8727','8728','A126','A127','8723','8724','3814'])
vars_rcl_cd = '|'.join(['IDRSSD', 'C968','C969','C970','C971','C972','C973','C974','C975'])
vars_rcr = '|'.join(['IDRSSD','B704','A222','3128','7204','7205','7206','A223'])
vars_rcs = '|'.join(['IDRSSD','B705','B706','B707','B708','B709','B710','B711',\
                     'B790','B791','B792','B793','B794','B795','B796',\
                     'B747','B748','B749','B750','B751','B752','B753'])  
vars_ri = '|'.join(['IDRSSD','4074','4230','4079','4080','4107','4073','4093','4010','4065','4115','4060','B488','B489','4069',\
                    '4020','4518','4508','4180','4185','4200','4340'])
vars_rib = '|'.join(['IDRSSD', '4230','4635','3123'])
#------------------------------------------
# Load the data
## define the dfs
df_info = pd.DataFrame()
df_rc = pd.DataFrame()
df_rcb = pd.DataFrame()
df_rcc = pd.DataFrame()
df_rce = pd.DataFrame()
df_rcg = pd.DataFrame()
df_rcl = pd.DataFrame()
df_rcl_cd = pd.DataFrame()
df_rcr = pd.DataFrame()
df_rcs = pd.DataFrame()
df_ri = pd.DataFrame()
df_rib = pd.DataFrame()
#------------------------------------------
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
#------------------------------------------
## Load rc data
for i in range(start,end):
    df_load = pd.read_csv((path_call + file_rc).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rc)]
    df_load['date'] = int('{}'.format(i))
    df_rc = df_rc.append(df_load)

### Merge RCFD and RCON cols
var_num = ['2170','3545','3548','0081','0071','2948','3210','2800']

for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rc['RC{}'.format(elem)] = df_rc.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1)  

df_rc['RC2200'] = df_rc.apply(lambda x: x['RCFN2200'] if not np.isnan(x['RCFN2200']) and round(x['RCFN2200']) != 0 else (x['RCON2200']), axis = 1)     
#------------------------------------------
## Load rcb data
for i in range(start,end):
    if i < 2009:
        df_load = pd.read_csv((path_call + file_rcb1).format(i,i), \
                 sep='\t',  skiprows = [1,2])
    else:
        df_load = pd.read_csv((path_call + file_rcb2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcb)]
    df_load['date'] = int('{}'.format(i))
    df_rcb = df_rcb.append(df_load)  

### Merge RCFD and RCON cols    
var_num = ['1771','1773','0213','1287','1754']
  
for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcb['RC{}'.format(elem)] = df_rcb.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1)      
#------------------------------------------
## Load rcc data
for i in range(start,end):
    df_load = pd.read_csv((path_call + file_rcc).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcc)]
    df_load['date'] = int('{}'.format(i))
    df_rcc = df_rcc.append(df_load)
    
### Merge RCFD and RCON cols    
var_num = ['1420','1460','1590','1797','2122']
  
for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcc['RC{}'.format(elem)] = df_rcc.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1)   
#------------------------------------------
## Load rce data
for i in range(start,end):
    df_load = pd.read_csv((path_call + file_rce).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rce)]
    df_load['date'] = int('{}'.format(i))
    df_rce = df_rce.append(df_load)
#------------------------------------------
## Load rcg data
for i in range(start,end):
    df_load = pd.read_csv((path_call + file_rcg).format(i,i), sep='\t',  skiprows = [1,2], engine = 'python', quoting=csv.QUOTE_NONE)
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcg)]
    df_load['date'] = int('{}'.format(i))
    df_rcg = df_rcg.append(df_load)
    
df_rcg.columns = ['IDRSSD', 'RCFDB557', 'RCONB557', 'date']
#------------------------------------------   
## Load rcl data  
for i in range(start,end):
    if i < 2009:
        df_load = pd.read_csv((path_call + file_rcl1).format(i,i), \
                 sep='\t',  skiprows = [1,2])
    else:
        df_load = pd.read_csv((path_call + file_rcl2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcl)]
    df_load['date'] = int('{}'.format(i))
    df_rcl = df_rcl.append(df_load)     

### Merge RCFD and RCON cols    
var_num = ['8725','8726','8727','8728','A126','A127','8723','8724']

for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcl['RC{}'.format(elem)] = df_rcl.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1)    
#------------------------------------------
## load rcl_cd data
for i in range(start + 5,end):
    if i < 2009:
        df_load = pd.read_csv((path_call + file_rcl1).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcl_cd)]
        df_load['date'] = int('{}'.format(i))
    else:
        df_load = pd.read_csv((path_call + file_rcl2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load2 = pd.read_csv((path_call + file_rcl3).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load['date'], df_load2['date'] = int('{}'.format(i)), int('{}'.format(i))
        df_load = df_load.merge(df_load2, on = ['IDRSSD', 'date'], how = 'outer')
        df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcl_cd)]
        df_load['date'] = int('{}'.format(i))

    df_rcl_cd = df_rcl_cd.append(df_load)

### Merge RCFD and RCON cols    
var_num = ['C968','C969','C970','C971','C972','C973','C974','C975']

for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcl_cd['RC{}'.format(elem)] = df_rcl_cd.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1) 
#------------------------------------------
## Load rcr data
for i in range(start,end):
    if i == 2014:
        df_load_rcfd = pd.read_csv((path_call + file_rcr2_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr2_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcri = pd.read_csv((path_call + file_rcri1).format(i,i), \
                 sep='\t',  skiprows = [1,2])
    
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcri, on = 'IDRSSD', how = 'left')
        df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcr)]
        df_load['date'] = int('{}'.format(i))
    elif i == 2015 or i == 2016:
        df_load_rcfd = pd.read_csv((path_call + file_rcr3_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr3_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcri = pd.read_csv((path_call + file_rcri2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcri, on = 'IDRSSD', how = 'left')
        df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcr)]
        df_load['date'] = int('{}'.format(i))
    elif i > 2016:
        df_load_rcfd = pd.read_csv((path_call + file_rcr4_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr4_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcri = pd.read_csv((path_call + file_rcri2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcri, on = 'IDRSSD', how = 'left')
        df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcr)]
        df_load['date'] = int('{}'.format(i))
    else:
        df_load_rcfd = pd.read_csv((path_call + file_rcr1_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr1_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
 
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcr)]
        df_load['date'] = int('{}'.format(i))
    
    df_rcr = df_rcr.append(df_load)

### Make Variables G641 (total RWA) and merge the two
df_rcr['RCFDG641'] = df_rcr.RCFDB704 - df_rcr.RCFDA222 - df_rcr.RCFD3128
df_rcr['RCONG641'] = df_rcr.RCONB704 - df_rcr.RCONA222 - df_rcr.RCON3128

var_num = ['G641','B704','A222','3128','A223']

for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcr['RC{}'.format(elem)] = df_rcr.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1) 
    
### Make compound variables for: '7204','7205','7206'
#### Drop variables not needed
vars_drop = '|'.join(['RCFW','RCOW'])
df_rcr = df_rcr.loc[:,~df_rcr.columns.str.contains(vars_drop)]

#### Transform RCFA and RCOA to float
vars_trans = '|'.join(['RCFA7','RCOA7'])
df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)] = df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)].apply(lambda x: x.str.strip('%').astype(float) / 100)

#### Make the variables
var_num = ['7204','7205','7206']

for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcr['RCF{}'.format(elem)] = df_rcr.apply(lambda x: x['RCFA{}'.format(elem)] if x.date >= 2014 else (x['RCFD{}'.format(elem)]), axis = 1)
    df_rcr['RCO{}'.format(elem)] = df_rcr.apply(lambda x: x['RCOA{}'.format(elem)] if x.date >= 2014 else (x['RCON{}'.format(elem)]), axis = 1) 

for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcr['RC{}'.format(elem)] = df_rcr.apply(lambda x: x['RCF{}'.format(elem)] if not np.isnan(x['RCF{}'.format(elem)]) and  round(x['RCF{}'.format(elem)]) != 0 else (x['RCO{}'.format(elem)]), axis = 1)     
#------------------------------------------
## Load rcs data    
for i in range(start,end):
    df_load = pd.read_csv((path_call + file_rcs).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rcs)]
    df_load['date'] = int('{}'.format(i))
    df_rcs = df_rcs.append(df_load) 

### Merge RCFD and RCON cols
var_num = ['B705','B706','B707','B708','B709','B710','B711','B790','B791','B792','B793','B794','B795','B796']

for i,elem in enumerate(var_num):
    '''Combines the RCFD and RCON variables into one variable. If RCFD is a number it takes the RCFD, RCON otherwise '''
    df_rcs['RC{}'.format(elem)] = df_rcs.apply(lambda x: x['RCFD{}'.format(elem)] if not np.isnan(x['RCFD{}'.format(elem)]) and  round(x['RCFD{}'.format(elem)]) != 0 else (x['RCON{}'.format(elem)]), axis = 1) 

### Make a total securitization and total loan sales variable
'''First fill the NaNs with zeros. Since we know that a NaN is a true zero'''
cols_nanfill = ['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711','RCB790','RCB791','RCB792','RCB793','RCB794','RCB795','RCB796']
df_rcs[cols_nanfill] = df_rcs[cols_nanfill].fillna(value = 0)
    
df_rcs['ls_sec_tot'] = df_rcs[['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711']].sum(axis = 1, skipna = True)
df_rcs['ls_nonsec_tot'] = df_rcs[['RCB790','RCB791','RCB792','RCB793','RCB794','RCB795','RCB796']]\
                            .sum(axis = 1, skipna = True)
df_rcs['ls_tot'] = df_rcs.ls_sec_tot + df_rcs.ls_nonsec_tot
#------------------------------------------
## Load ri data    
for i in range(start,end):
    df_load = pd.read_csv((path_call + file_ri).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_ri)]
    df_load['date'] = int('{}'.format(i))
    df_ri = df_ri.append(df_load) 
#------------------------------------------
## Load rib data    
for i in range(start,end):
    df_load_ribi = pd.read_csv((path_call + file_ribi).format(i,i), sep='\t',  skiprows = [1,2])
    df_load_ribii = pd.read_csv((path_call + file_ribii).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load_ribi.merge(df_load_ribii, on = 'IDRSSD', how = 'left')
    df_load = df_load.loc[:,df_load.columns.str.contains(vars_rib)]
    df_load['date'] = int('{}'.format(i))
    df_rib = df_rib.append(df_load) 

#------------------------------------------
# Merge data and save the raw file 
df_raw = df_rc.merge(df_rcc, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcb, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rce, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcg, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcl, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcl_cd, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcr, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcs, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_ri, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rib, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_info, on = ['IDRSSD', 'date'], how = 'left')

df_raw.to_csv('df_assetcomp_raw.csv')

