 di# Balance panel
''' This function takes a pandas dataframe that is structured as a unbalanced
    panel and balances it. 
    
    Clean the data beforehand!
    
    Input: pandas dataframe, name entity column, name time column, multi-index
    Output: Balanced dataframe with multi-index
    
    Packages needed: Pandas
    '''

#Import packages
import pandas as pd

def balancePanel(df, iden = 'IDRSSD', t = 'date', mindex = False):
    # Check whether the df is not empty
    if df.shape == (0,0):
        raise Exception('Error: the input dataframe is empty')
    
    # Prelims
    total_t = df[t].unique().shape[0] 
    
    # Balance the panel set    
    if not mindex:
        unique_IDRSSD = df[iden].unique().astype(int)
        load_IDRSSD = []
        
        for i,id in enumerate(unique_IDRSSD):
            if not(df[df[iden] == id].shape[0] == total_t):
                load_IDRSSD.append(id)
        
        df_balanced = df[~df[iden].isin(load_IDRSSD)]  
    else:
        pass #TODO
       
    ## Add check whether the balancing is good
    if df_balanced.shape[0] == 0:
        raise Exception('Error: the output dataframe is empty')
    
    if (df_balanced[iden].unique().shape[0] * df_balanced[t].unique().shape[0]) != df_balanced.shape[0]:
        raise Exception('Error: Balancing of the dataframe failed')
         
    return (df_balanced)