#------------------------------------------
# Setup dataset US Call reports
# Mark van der Plaat
# August 2019; data update: Mar 2020  

''' 
    This document sets up the dataset for further analyses for working paper #1

    Data used: US Call reports 2001-2019 year-end
    Only insured, commercial banks are taken
    ''' 
#------------------------------------------------------------
# Import Packages
#------------------------------------------------------------
    
import pandas as pd
import numpy as np 

import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

import os
#os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')
os.chdir(r'D:\RUG\PhD\Materials_papers\1_Working_paper_loan_sales')

import csv

#------------------------------------------------------------
# Set prelims
#------------------------------------------------------------

start = 2001
end = 2020
num_cores = mp.cpu_count()

#------------------------------------------------------------
# Set file paths/names and variable names
#------------------------------------------------------------

#------------------------------------------
# Set file paths

#path_info = r'X:/My Documents/Data/Data_call_reports_fed'
#path_call = r'X:/My Documents/Data/Data_call_reports_FFIEC2'
path_info = r'D:/RUG/Data/Data_call_reports_fed'
path_call = r'D:/RUG/Data/Data_call_reports_FFIEC2'

#------------------------------------------
# Set filenames
# NOTE: File names change every once in a while, leading to multiple filenames

file_info1 = r'/call0{}12.xpt'
file_info2 = r'/call{}12.xpt'

file_por = r'/{}/FFIEC CDR Call Bulk POR 1231{}.txt'

file_rc = r'/{}/FFIEC CDR Call Schedule RC 1231{}.txt'

file_rcb1 = r'/{}/FFIEC CDR Call Schedule RCB 1231{}.txt'
file_rcb2 = r'/{}/FFIEC CDR Call Schedule RCB 1231{}(1 of 2).txt'

file_rcc = r'/{}/FFIEC CDR Call Schedule RCCI 1231{}.txt'

file_rce = r'/{}/FFIEC CDR Call Schedule RCE 1231{}.txt'

file_rcg = r'/{}/FFIEC CDR Call Schedule RCG 1231{}.txt'

file_rcl1 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}.txt'
file_rcl2_1 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(1 of 2).txt'  # From 2009
file_rcl2_2 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(2 of 2).txt'  # From 2009

file_rcn1 = r'/{}/FFIEC CDR Call Schedule RCN 1231{}.txt' 
file_rcn2_1 = r'/{}/FFIEC CDR Call Schedule RCN 1231{}(1 of 2).txt'  # From 2011
file_rcn2_2 = r'/{}/FFIEC CDR Call Schedule RCN 1231{}(2 of 2).txt'  # From 2011

file_rcr1_rcfd = r'/{}/FFIEC CDR Call Schedule RCR 1231{}(1 of 2).txt'
file_rcr2_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 2).txt' #2014
file_rcr3_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 3).txt' #from 2015
file_rcr4_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 4).txt' #from 2017
file_rcr1_rcon = r'/{}/FFIEC CDR Call Schedule RCR 1231{}(2 of 2).txt'
file_rcr2_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 2).txt' #2014
file_rcr3_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 3).txt' #from 2015
file_rcr3b_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(3 of 3).txt' #from 2015
file_rcr4_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 4).txt' #from 2017
file_rcr4b_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(3 of 4).txt' #from 2017

file_rcria = r'/{}/FFIEC CDR Call Schedule RCRIA 1231{}.txt' #2014
file_rcrib = r'/{}/FFIEC CDR Call Schedule RCRIB 1231{}.txt' #2014
file_rcri2 = r'/{}/FFIEC CDR Call Schedule RCRI 1231{}.txt' #from 2015

file_rcs = r'/{}/FFIEC CDR Call Schedule RCS 1231{}.txt'

file_ri = r'/{}/FFIEC CDR Call Schedule RI 1231{}.txt'

file_ribi = r'/{}/FFIEC CDR Call Schedule RIBI 1231{}.txt'
file_ribii = r'/{}/FFIEC CDR Call Schedule RIBII 1231{}.txt'

file_su = r'/{}/FFIEC CDR Call Schedule SU 1231{}.txt' #from 2017 only

#------------------------------------------
# Set variables needed per csv
# NOTE: Most variable names have multiple prefixes. We use a work around

## From balance sheet
vars_info = ['RSSD9001','RSSD9999','RSSD9048','RSSD9424','RSSD9170','RSSD9210','RSSD9364']
vars_rc = '|'.join(['IDRSSD','2170','3545','3548','2200','0081','0071','3210','2948',\
                    '2800','B993','B995','3200']) 
vars_rcb = '|'.join(['IDRSSD','1771','1773','0213','1287','1754'])
vars_rcc = '|'.join(['IDRSSD','1410','1415','1420','1797','1460','1288','1590','1766',\
                     '2122','1590','F158','F159','B538','B539','2011','K137','K207',\
                     '5367','5368','1480','F160','F161','1763','1764','2123','1616',\
                     'F576','K158','K159','K160','K161','K162','K163','K164','K165',\
                     'K256']) 
vars_rce = '|'.join(['IDRSSD','B549','B550','6648','2604','J473','J474','B535','2081'])
vars_rcg = '|'.join(['IDRSSD', 'B557'])
vars_rcl = '|'.join(['IDRSSD','3814','3815','3816','6550','3817','3818',\
                     '3819','3821','3411','3428','3433','A534','A535',\
                     '3430','5591','A126','A127','8723','8724','8725',\
                     '8726','8727','8728','F164','F165','J477','J478',\
                     'J457','J458','J459'] + ['C{}'.format(i) for i in range(968,975+1)])
vars_rcn = '|'.join(['IDRSSD','2759','2769','3492','3493','3494','3495',\
                     '5398','5399','5400','5401','5402','5403',\
                     '3499','3500','3501','3502','3503','3504',\
                     '5377','5378','5379','5380','5381','5382',\
                     '1594','1597','1583','1251','1252','1253',\
                     '1254','1255','1256','B575','B576','B577',\
                     'B578','B579','B580','5389','5390','5391',\
                     '5459','5460','5461','1257','1258','1259',\
                     '1271','1272','1791','F172','F174','F176',\
                     'F173','F175','F177','C236','C237','C229',\
                     'C238','C239','C230','F178','F180','F182',\
                     'F179','F181','F183','K213','K214','K215',\
                     'K216','K217','K218','F166','F167','F168',\
                     'F169','F170','F171','1406','1407','1408',\
                     'B834','B835','B836','1658','1659','1661',\
                     'K105','K106','K107','K108','K109','K110',\
                     'F661','F662','F663','K111','K112','K113',\
                     'K114','K115','K116','K117','K118','K119',\
                     'K257','K258','K259','K126','K127','K128',\
                     'K120','K121','K122','K123','K124','K125',\
                     '1606','1607','1608']) # Every first is <89; second >89; third nonaccrual. Last three at totals for FFIEC031 > 2011
vars_rcr = '|'.join(['IDRSSD','B704','A222','3128','7204','7205','7206','A223',\
                     'B645','B650','B655','B660','B664','B669','2243','B676',\
                     'B682','B687','A167','G592','D992','D998','G607','G613',\
                     'S516','G619','S526','G625','S541','S542','S549'])
vars_rcs = '|'.join(['IDRSSD','B705','B706','B707','B708','B709','B710','B711',\
                     'B790','B791','B792','B793','B794','B795','B796',\
                     'B747','B748','B749','B750','B751','B752','B753',\
                     'B754','B755','B756','B757','B758','B759','B760',\
                     'B712','B713','B714','B715','B716','B717','B718',\
                     'B719','B720','B721','B722','B723','B724','B725',\
                     'B797','B798','B799','B800','B801','B802','B803',\
                     'C393','C394','C395','C396','C397','C398','C399',\
                     'C400','C401','C402','C403','C404','C405','C406',\
                     'HU09','HU10','HU11','HU12','HU13','HU14','HU15',\
                     'B740','B741','B742','B743','B744','B745','B746',\
                     'B776','B777','B778','B779','B780','B781','B782',\
                     'A249','A250'])  

## From income statement
vars_ri = '|'.join(['IDRSSD','4074','4230','4079','4080','4107','4073','4093',\
                    '4010','4065','4115','4060','B488','B489','4069','4020',\
                    '4518','4508','4180','4185','4200','4340','4150','4301',\
                    '3521','3196','B493','5416','5415','B496'])
vars_rib = '|'.join(['IDRSSD', '4230','4635','4605','3123'])

## Supplemental information
vars_su = '|'.join(['IDRSSD','FT08','FT10'])

#------------------------------------------------------------
# Set functions
#------------------------------------------------------------
def loadGeneral(i, file, var_list):
    ''' A General function for loading call reports data, no breaks '''
    global path_call
    
    df_load = pd.read_csv((path_call + file).format(i,i), sep='\t',  skiprows = [1,2])
    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])
    
def loadGeneralAlt(i, file, var_list):
    ''' A General function for loading call reports data, no breaks '''
    global path_call
    
    df_load = pd.read_csv((path_call + file).format(i,i), sep='\t',  skiprows = [1,2],\
                          engine = 'python', quoting=csv.QUOTE_NONE)
    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])
    
def loadGeneralOneBreak(i, file1, file2, var_list, break_point):
    ''' A General function for loading call reports data, no breaks '''
    global path_call
    
    if i < break_point:
        df_load = pd.read_csv((path_call + file1).format(i,i), sep='\t',  skiprows = [1,2])
    else:
        df_load = pd.read_csv((path_call + file2).format(i,i), sep='\t',  skiprows = [1,2])

    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])
 
def loadInfo(i,break_point):
    ''' Function to load the info data'''
    global path_info, file_info1, file_info2, vars_info
    
    if i < break_point:
        df_load = pd.read_sas((path_info + file_info1).format(i))
    else:
        df_load = pd.read_sas((path_info + file_info2).format(i))
        
    return(df_load[vars_info])

def loadRCL(i):
    ''' A General function for loading call reports data, no breaks '''
    global path_call
    
    if i < 2009:
        df_load = pd.read_csv((path_call + file_rcl1).format(i,i), sep='\t',  skiprows = [1,2])
    else:
        df_load_1 = pd.read_csv((path_call + file_rcl2_1).format(i,i), sep='\t',  skiprows = [1,2])
        df_load_2 = pd.read_csv((path_call + file_rcl2_2).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load_1.merge(df_load_2, on = 'IDRSSD', how = 'left')

    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rcl + '|date')])    
    
def loadRCN(i):
    ''' A General function for loading call reports data, no breaks '''
    global path_call
    
    if i < 2011:
        df_load = pd.read_csv((path_call + file_rcn1).format(i,i), sep='\t',  skiprows = [1,2])
    else:
        df_load_1 = pd.read_csv((path_call + file_rcn2_1).format(i,i), sep='\t',  skiprows = [1,2])
        df_load_2 = pd.read_csv((path_call + file_rcn2_2).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load_1.merge(df_load_2, on = 'IDRSSD', how = 'left')

    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rcn + '|date')])    
    
def loadRCR(i):
    if i == 2014:
        df_load_rcfd = pd.read_csv((path_call + file_rcr2_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr2_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcria = pd.read_csv((path_call + file_rcria).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcrib = pd.read_csv((path_call + file_rcrib).format(i,i), \
                 sep='\t',  skiprows = [1,2])
    
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcria, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcrib, on = 'IDRSSD', how = 'left')
        
    elif i == 2015 or i == 2016:
        df_load_rcfd = pd.read_csv((path_call + file_rcr3_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr3_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcon2 = pd.read_csv((path_call + file_rcr3b_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcri = pd.read_csv((path_call + file_rcri2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcon2, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcri, on = 'IDRSSD', how = 'left')
        
    elif i > 2016:
        df_load_rcfd = pd.read_csv((path_call + file_rcr4_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr4_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcon2 = pd.read_csv((path_call + file_rcr4b_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcri = pd.read_csv((path_call + file_rcri2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcon2, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcri, on = 'IDRSSD', how = 'left')
        
    else:
        df_load_rcfd = pd.read_csv((path_call + file_rcr1_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr1_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
 
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        
    df_load['date'] = int('{}'.format(i))
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rcr + '|date')])
    
def loadRIB(i):
    df_load_ribi = pd.read_csv((path_call + file_ribi).format(i,i), sep='\t',  skiprows = [1,2])
    df_load_ribii = pd.read_csv((path_call + file_ribii).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load_ribi.merge(df_load_ribii, on = 'IDRSSD', how = 'left')
    df_load['date'] = int('{}'.format(i))
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rib+ '|date')])  
    
def combineVars(data, elem):
    ''' Function to combine RCFD and RCON into one variable '''
    data['RC' + elem] = data['RCFD' + elem].fillna(data['RCON' + elem])
    
    return(data['RC' + elem])

def combineVarsAlt(data, elem):
    data['RC' + elem] = data['RCF' + elem].fillna(data['RCO' + elem])
    
    return(data['RC' + elem])

#------------------------------------------------------------
# Load Data
#------------------------------------------------------------

#------------------------------------------
## Load info data 
if __name__ == '__main__':
    df_info = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadInfo)(i,10) for i in range(start - 2000, end - 2000)))

### Rename RSSD9001 and RSSD9999
df_info.rename(columns = {'RSSD9001':'IDRSSD', 'RSSD9999':'date'}, inplace = True)

### Change date to only the year
df_info.date = (df_info.date.round(-4) / 1e4).astype(int)

#------------------------------------------
## Load rc data
if __name__ == '__main__':
    df_rc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rc, vars_rc) for i in range(start, end)))

### Merge RCFD and RCON cols
var_num = ['2170','3545','3548','0081','0071','2948','3210','2800']

if __name__ == '__main__':
    df_rc_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rc, elem) for elem in var_num), axis = 1)

cols_remove =  [col for col in df_rc.columns if not col[4:] in var_num]     
df_rc = pd.concat([df_rc[cols_remove], df_rc_combvars], axis = 1)    

df_rc['RC2200'] = df_rc.loc[:,['RCFN2200','RCON2200']].sum(axis = 1)
 
#------------------------------------------
## Load rcb data
if __name__ == '__main__':
    df_rcb = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneralOneBreak)(i, file_rcb1, file_rcb2, vars_rcb, 2009) for i in range(start, end)))

### Merge RCFD and RCON cols    
var_num = ['1771','1773','0213','1287','1754']

if __name__ == '__main__':
    df_rcb_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rcb, elem) for elem in var_num), axis = 1)

cols_remove =  [col for col in df_rcb.columns if not col[4:] in var_num]      
df_rcb = pd.concat([df_rcb[cols_remove], df_rcb_combvars], axis = 1)    
  
#------------------------------------------
## Load rcc data
if __name__ == '__main__':
    df_rcc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcc, vars_rcc) for i in range(start, end)))
    
### Merge RCFD and RCON cols    
var_num = ['1420','1460','1590','1797','2122','B538','B539','2011','K137','K207',\
           '5367','5368','F158','F159','F160','F161','1763','1764','2123','1616',\
           'K165']

if __name__ == '__main__':
    df_rcc_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rcc, elem) for elem in var_num), axis = 1)

cols_remove =  [col for col in df_rcc.columns if not col[4:] in var_num]      
df_rcc = pd.concat([df_rcc[cols_remove], df_rcc_combvars], axis = 1)   

    
#------------------------------------------
## Load rce data
if __name__ == '__main__':
    df_rce = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rce, vars_rce) for i in range(start, end)))

#------------------------------------------
## Load rcg data
if __name__ == '__main__':
    df_rcg = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneralAlt)(i, file_rcg, vars_rcg) for i in range(start, end)))
    
df_rcg.columns = ['IDRSSD', 'RCFDB557', 'RCONB557', 'date']

df_rcg['RCB557'] = combineVars(df_rcg, 'B557')

#------------------------------------------
## Load rcl data
if __name__ == '__main__':
    df_rcl = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCL)(i) for i in range(start, end)))
    
### Merge RCFD and RCON cols
var_num = [elem for elem in vars_rcl.split('|') if np.sum(df_rcl.columns.str.contains(elem)) == 2]

if __name__ == '__main__':
    df_rcl_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rcl, elem) for elem in var_num), axis = 1)

cols_remove = [col for col in df_rcl.columns if not col[4:] in var_num] 
df_rcl = pd.concat([df_rcl[cols_remove], df_rcl_combvars], axis = 1)   

#------------------------------------------
## Load rcn data
if __name__ == '__main__':
    df_rcn = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCN)(i) for i in range(start, end)))
    
### Merge RCFD and RCON cols
var_num = [elem for elem in vars_rcn.split('|') if np.sum(df_rcn.columns.str.contains(elem)) == 2]

if __name__ == '__main__':
    df_rcn_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rcn, elem) for elem in var_num), axis = 1)

cols_remove = [col for col in df_rcn.columns if not col[4:] in var_num] 
df_rcn = pd.concat([df_rcn[cols_remove], df_rcn_combvars], axis = 1)   

#------------------------------------------
## Load rcr data
if __name__ == '__main__':
    df_rcr = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCR)(i) for i in range(start, end)))

### Make Variables G641 (total RWA) and merge the two
df_rcr['RCFDG641'] = df_rcr.RCFDB704 - df_rcr.RCFDA222 - df_rcr.RCFD3128
df_rcr['RCONG641'] = df_rcr.RCONB704 - df_rcr.RCONA222 - df_rcr.RCON3128

var_num = ['G641','B704','A222','3128','A223',\
           'B645','B650','B655','B660','B664','B669','2243','B676',\
           'B682','B687','A167','G592','D992','D998','G607','G613',\
           'S516','G619','S526','G625','S541','S542','S549']

if __name__ == '__main__':
    df_rcr_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rcr, elem) for elem in var_num), axis = 1)
 
cols_remove = [col for col in df_rcr.columns if not col[4:] in var_num]     
df_rcr = pd.concat([df_rcr[cols_remove], df_rcr_combvars], axis = 1)   
   
### Make compound variables for: '7204','7205','7206'
#### Drop variables not needed
vars_drop = '|'.join(['RCFW','RCOW'])
df_rcr = df_rcr.loc[:,~df_rcr.columns.str.contains(vars_drop)]

#### Transform RCFA and RCOA to float
vars_trans = '|'.join(['RCFA7','RCOA7'])
df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)] = df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)].apply(lambda x: x.str.strip('%').astype(float) / 100)

#### Make the variables
var_num = ['7204','7205','7206']

for elem in var_num:
    df_rcr['RCF{}'.format(elem)] = df_rcr.loc[:,'RCFD{}'.format(elem)].fillna(df_rcr.loc[:,'RCFA{}'.format(elem)])
    df_rcr['RCO{}'.format(elem)] = df_rcr.loc[:,'RCON{}'.format(elem)].fillna(df_rcr.loc[:,'RCOA{}'.format(elem)])

if __name__ == '__main__':
    df_rcr_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVarsAlt)(df_rcr, elem) for elem in var_num), axis = 1)

cols_remove = [col for col in df_rcr.columns if not col[4:] in var_num]      
df_rcr = pd.concat([df_rcr[cols_remove], df_rcr_combvars], axis = 1)

#------------------------------------------
## Load rcs data 
if __name__ == '__main__':
    df_rcs = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcs, vars_rcs) for i in range(start, end)))
    
var_num = ['B705','B706','B707','B708','B709','B710','B711',\
           'B790','B791','B792','B793','B794','B795','B796',\
           'B712','B713','B714','B715','B716','B717','B718',\
           'B719','B720','B721','B722','B723','B724','B725',\
           'B797','B798','B799','B800','B801','B802','B803',\
           'C393','C394','C395','C396','C397','C398','C399',\
           'C400','C401','C402','C403','C404','C405','C406',\
           'HU09','HU15','B740','B741','B742','B743','B744',\
           'B745','B746','B776','B777','B778','B779','B780',\
           'B781','B782','A249','A250']

if __name__ == '__main__':
    df_rcs_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_rcs, elem) for elem in var_num), axis = 1)

cols_remove =  [col for col in df_rcs.columns if not col[4:] in var_num]      
df_rcs = pd.concat([df_rcs[cols_remove], df_rcs_combvars], axis = 1)   
                 
#------------------------------------------
## Load ri data
if __name__ == '__main__':
    df_ri = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_ri, vars_ri) for i in range(start, end)))
    
#------------------------------------------
## Load rib data
if __name__ == '__main__':
    df_rib = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRIB)(i) for i in range(start, end)))    
    
#------------------------------------------
## Load su data
if __name__ == '__main__':
    df_su = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_su, vars_su) for i in range(2017, end)))

#------------------------------------------
# Merge data and save the raw file 
df_raw = df_rc.merge(df_rcc, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcb, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rce, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcg, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcl, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcn, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcr, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rcs, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_ri, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_rib, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_su, on = ['IDRSSD', 'date'], how = 'left')
df_raw = df_raw.merge(df_info, on = ['IDRSSD', 'date'], how = 'left')

''' Turn on if you want to check
#------------------------------------------
# Check RIAD 4230 (are two variables with that name)
sum(df_raw.RIAD4230_y.eq(df_raw.RIAD4230_x) * 1) # two entries are not identical..

## Check the different entries
df_raw.loc[~df_raw.RIAD4230_y.eq(df_raw.RIAD4230_x), ['RIAD4230_x','RIAD4230_y']]
#There are some differences, but they are minor. Keep RIAD4320_y and rename
'''
 
# Drop and rename
df_raw.drop('RIAD4230_x', axis = 1, inplace = True)
df_raw.rename(columns = {'RIAD4230_y':'RIAD4230'}, inplace = True)

#------------------------------------------
# Drop double RIAD4605
df_raw.drop('RIAD4605_x', axis = 1, inplace = True)
df_raw.rename(columns = {'RIAD4605_y':'RIAD4605'}, inplace = True)

df_raw.to_csv('Data\df_wp1_raw.csv', index = False)

