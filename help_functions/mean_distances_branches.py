# Mean, Median, STD branches (Summary of deposits)
''' This function reads in the summary of deposits data and the mean, median or
    standard deviation of the distance between branches and the head quarters
    per year. 

    Packages needed: Pandas, Geocoder, Numpy, Numba, Math
    '''
    
# Import Packages
import pandas as pd
import numpy as np
import geocoder

import os
os.chdir(r'X:\My Documents\Data\Data_SOD')

#--------------------------------------------------
# Prelim
begin_year = 2001
end_year = 2019

#--------------------------------------------------

# Function
from numba import njit, jit

@jit
def readSODFiles(use_cols,begin_year,end_year):
    '''This method reads in all the data from the Summary of Deposits and returns
        a dataframe with the selected columns only. Requires numba,jit'''
    path = r'X:/My Documents/Data/Data_SOD/'
    df_sod = pd.DataFrame()
    
    #Read data  
    for i in range(begin_year,end_year):
        df_load = pd.read_csv(path + 'ALL_{}.csv'.format(i), encoding = 'ISO-8859-1', usecols = use_cols)
        df_sod = df_sod.append(df_load,ignore_index = True)
    
    return (df_sod)

use_cols = ['YEAR','RSSDID','ADDRESBR','CITYBR','STALPBR','ADDRESS','CITY',\
            'STALP','BKMO','SIMS_LATITUDE','SIMS_LONGITUDE'] 
df_sod = readSODFiles(use_cols,begin_year,end_year)

#--------------------------------------------------

@jit
def lookupLatLong(x):
    ''' This method looks up the addresses of all branches and head quarters in the SOD.
    To speed up the process, we will only look up the coordinates for the branches that have missing
    information for latitude and longitude, but have an address.'''
    x_nan = x[np.isnan(x.SIMS_LATITUDE)]
    x_nan.reset_index(inplace = True, drop = True)
    
    res = pd.DataFrame(x_nan[['YEAR','RSSDID','ADDRESBR','CITYBR','STALPBR']],\
                       columns = ['YEAR','RSSDID','ADDRESBR','CITYBR','STALPBR',\
                                  'lat_arcgis','long_arcgis'])
    address = (x_nan['ADDRESBR'] + ', ' + x_nan['CITYBR'] + ' ' + x_nan['STALPBR'])
    
    for i in range(x_nan.shape[0]):
        g = geocoder.arcgis(address[i])
        res.lat_arcgis[i], res.long_arcgis[i] = g.lat, g.lng
    
    return(res)

geo_list = lookupLatLong(df_sod)
geo_list.to_csv('SOD_all_coordinates.csv')

# Add geo_list to df_sod
df_sod = df_sod.merge(geo_list, on = ['YEAR','RSSDID','ADDRESBR','CITYBR','STALPBR'],\
             how = 'left', indicator = True)

df_sod['branch_lat'] = df_sod[['SIMS_LATITUDE','lat_arcgis']].sum(axis=1)
df_sod['branch_lng'] = df_sod[['SIMS_LONGITUDE','long_arcgis']].sum(axis=1)

#-------------------------------------------------
# Make two variables that are the lat and long for the head quarters
## First drop _merge and coordinates that are zero.
df_sod.drop(['_merge'], axis = 1, inplace = True)

## Restrict the latitude to strictly positive and longitude to strictly negative
df_sod = df_sod[(df_sod.branch_lat > 0.0) & (df_sod.branch_lng < 0.0)]

df_head = df_sod[df_sod.BKMO == 1]
df_head.rename(columns = {'branch_lat':'head_lat','branch_lng':'head_lng'}, inplace = True)
df_sod_complete = df_sod.merge(df_head[['YEAR','RSSDID','head_lat','head_lng']], on = ['YEAR','RSSDID'], how = 'left', indicator = True)

#Drop duplicates and _merge and where both lat and long are 0.0
df_sod_complete.drop(['_merge'], axis = 1, inplace = True)
df_sod_complete.drop_duplicates(subset = ['YEAR','RSSDID','ADDRESBR','CITYBR','STALPBR'], inplace = True)
df_sod_complete.reset_index(inplace = True)

df_sod_complete.to_csv('SOD_all_coordinates_trans.csv')

#------------------------------------------------
# Calculate the distance

from math import sin, cos, sqrt, atan2, radians

@jit(nopython = True, fastmath = True)
def distanceCalculator(lat1, lon1, lat2, lon2):
    ''' This method calculates the distance in km between two points on the map (calculated
        by the latitude and longitude). It uses Numby to speed up the process. 
        
        Calculation method used: Haversine formula
    '''
    # Define the method that calculates the distance
    def haversine(lat1, lon1, lat2, lon2):
        R = 6372.8  # Earth radius in km
    
        phi1, phi2 = radians(lat1), radians(lat2) 
        dphi       = radians(lat2 - lat1)
        dlambda    = radians(lon2 - lon1)
        
        a = sin(dphi/2)**2 + \
            cos(phi1)*cos(phi2)*sin(dlambda/2)**2
        
        return 2*R*atan2(sqrt(a), sqrt(1 - a))
    
    # Setup the results vector
    res = np.empty(len(lat1))

    # Loop over the vectors
    for i in range(len(lat1)):
        res[i] = haversine(lat1[i], lon1[i], lat2[i], lon2[i])
        
    return (res)

df_sod_complete['distance'] = distanceCalculator(np.array(df_sod_complete.branch_lat),\
                                                 np.array(df_sod_complete.branch_lng),\
                                                 np.array(df_sod_complete.head_lat),\
                                                 np.array(df_sod_complete.head_lng))

# Save the df
df_sod_complete.to_csv('SOD_all_distances.csv')


        
