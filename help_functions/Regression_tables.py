#------------------------------------------
# Method for making nice regression tables
# Mark van der Plaat
# October 2019 

'''These methods are taken form summary2 statsmodels. The last one retuns a simple
    dataframe that can be transformed to excel, csv or latex'''

import pandas as pd
import numpy as np
from statsmodels.compat.python import reduce, lrange

#------------------------------------------------
def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, float_format="%.4f"):
    '''create a summary table of parameters from results instance

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : string or None
        optional name for the endogenous variable, default is "y"
    xname : list of strings or None
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.
    float_format : string
        float formatting options (e.g. ".3g")

    Returns
    -------
    params_table : SimpleTable instance
    '''
    from linearmodels.panel.results import PanelEffectsResults
    from linearmodels.panel.results import RandomEffectsResults 
    from linearmodels.panel.results import PanelResults
    res_tuple = (PanelEffectsResults,PanelResults,RandomEffectsResults)

    if isinstance(results, tuple):
        results, params, std_err, tvalues, pvalues, conf_int = results
   
   #+4 : 
   # I added Panel results whose some attributes name are different.
   # So I modified the code as follows.

    elif isinstance(results,res_tuple):
        bse = results.std_errors
        tvalues = results.tstats
        conf_int = results.conf_int(1-alpha)
    else:
        bse = results.bse
        tvalues = results.tvalues
        conf_int = results.conf_int(alpha) 
    params = results.params
    pvalues = results.pvalues

    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, conf_int])
    data = pd.DataFrame(data)

    if use_t:
        data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
    else:
        data.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']

    if not xname:
        # data.index = results.model.exog_names
        try:
            data.index = results.model.exog_names
        except (AttributeError):
            data.index = results.model.exog.vars
    else:
        data.index = xname

    return data

#------------------------------------------------
def _col_params(result, float_format='%.4f', stars=True,show='t'):
    '''Stack coefficients and standard errors in single column
    '''

 # I add the parameter 'show' equals 't' to display tvalues by default,
 # 'p' for pvalues and 'se' for std.err.
    
    # Extract parameters
    res = summary_params(result)
   
    # Format float
    # Note that scientific number will be formatted to  'str' type though '%.4f'

    for col in res.columns[:3]:
        res[col] = res[col].apply(lambda x: float_format % x)
    res.iloc[:,3] = np.around(res.iloc[:,3],4)
    
    # Significance stars
    # .ix method will be deprecated,so .loc has been used.

    if stars:
        idx = res.iloc[:, 3] < .1
        res.loc[res.index[idx], res.columns[0]] += '*'
        idx = res.iloc[:, 3] < .05
        res.loc[res.index[idx], res.columns[0]] += '*'
        idx = res.iloc[:, 3] < .01
        res.loc[res.index[idx], res.columns[0]] += '*'

    # Std.Errors or tvalues or  pvalues in parentheses
    res.iloc[:,3] = res.iloc[:,3].apply(lambda x: float_format % x) # pvalues to str
    res.iloc[:, 1] = '(' + res.iloc[:, 1] + ')'
    res.iloc[:, 2] = '(' + res.iloc[:, 2] + ')'
    res.iloc[:, 3] = '(' + res.iloc[:, 3] + ')'

    # Stack Coefs and Std.Errors or pvalues
    if show is 't':
        res = res.iloc[:,[0,2]]
    elif show is 'se':
        res = res.iloc[:, :2]
    elif show is 'p':
        res = res.iloc[:,[0,3]]
    res = res.stack()
    res = pd.DataFrame(res)
    try:
        res.columns = [str(result.model.endog_names)]
    except (AttributeError):
        res.columns = result.model.dependent.vars #for PanelOLS
    
   # I added the index name transfromation function 
   # to deal with MultiIndex and single level index.
#----------------------------------------------
def _make_unique(list_of_names):
    if len(set(list_of_names)) == len(list_of_names):
        return list_of_names
    # pandas does not like it if multiple columns have the same names
    from collections import defaultdict
    dic_of_names = defaultdict(list)
    for i,v in enumerate(list_of_names):
        dic_of_names[v].append(i)
    for v in  dic_of_names.values():
        if len(v)>1:
            c = 0
            for i in v:
                c += 1
                list_of_names[i] += '_%i' % c
    return list_of_names

#-------------------------------------------------
def _col_info(result, more_info=None):
   
    '''Stack model info in a column
    '''
    model_info = summary_model(result)
    default_info_ = OrderedDict()
    default_info_['Model:'] = lambda x: x.get('Model:')
    default_info_['No. Observations:'] = lambda x: x.get('No. Observations:')
    default_info_['R-squared:'] = lambda x: x.get('R-squared:')
    default_info_['Adj. R-squared:'] = lambda x: x.get('Adj. R-squared:')                    
    default_info_['Pseudo R-squared:'] = lambda x: x.get('Pseudo R-squared:')
    default_info_['F-statistic:'] = lambda x: x.get('F-statistic:')
    default_info_['Covariance Type:'] = lambda x: x.get('Covariance Type:')
    default_info_['Eeffects:'] = lambda x: x.get('Effects:')
    default_info_['Covariance Type:'] = lambda x: x.get('Covariance Type:')

    default_info = default_info_.copy()
    for k,v in default_info_.items():
        if v(model_info):
            default_info[k] = v(model_info)
        else:
            default_info.pop(k) # pop the item whose value is none.
            
    if more_info is None:
        more_info = default_info
    else:
        if not isinstance(more_info,list):
            more_info = [more_info]
        for i in more_info:
            try:
                default_info[i] = getattr(result,i)
            except (AttributeError, KeyError, NotImplementedError) as e:
                raise e
        more_info = default_info
    try:
        out = pd.DataFrame(more_info, index=[result.model.endog_names]).T
    except (AttributeError):
        out = pd.DataFrame(more_info, index=result.model.dependent.vars).T
    return out
#----------------------------------------------
def regressionTables(results,stars = True, float_format='%.4f',show='se'):
    cols = [_col_params(x, stars=stars, float_format=float_format,show=show) for x in
        [results]]

#-----------------------------------------------
    def summary_col(results, float_format='%.4f', model_names=[], stars=True,
                more_info=None, regressor_order=[],show='t',title=None): 
        if not isinstance(results, list):
            results = [results]
    
        cols = [_col_params(x, stars=stars, float_format=float_format,show=show) for x in
                results]
    
        # Unique column names (pandas has problems merging otherwise)
        if model_names:
            colnames = _make_unique(model_names)
        else:
            colnames = _make_unique([x.columns[0] for x in cols])
        for i in range(len(cols)):
            cols[i].columns = [colnames[i]]
    
        merg = lambda x, y: x.merge(y, how='outer', right_index=True,
                                    left_index=True)
        summ = reduce(merg, cols)
    
        # if regressor_order:
        if not regressor_order:
            regressor_order = ['const']
        
        varnames = summ.index.get_level_values(0).tolist()
        ordered = [x for x in regressor_order if x in varnames]
        unordered = [x for x in varnames if x not in regressor_order + ['']]
    
        # Note: np.unique can disrupt the original order  of list 'unordered'.
        # Then pd.Series().unique()  works well.
    
        # order = ordered + list(np.unique(unordered))
        order = ordered + list(pd.Series(unordered).unique())
    
        f = lambda idx: sum([[x + 'coef', x + 'stde'] for x in idx], [])
        # summ.index = f(np.unique(varnames))
        summ.index = f(pd.Series(varnames).unique())
        summ = summ.reindex(f(order))
        summ.index = [x[:-4] for x in summ.index]
    
        idx = pd.Series(lrange(summ.shape[0])) % 2 == 1
        summ.index = np.where(idx, '', summ.index.get_level_values(0))
        summ = summ.fillna('')
        
        # add infos about the models.
        cols = [_col_info(x,more_info=more_info) for x in results]
        
        # use unique column names, otherwise the merge will not succeed
        for df , name in zip(cols, _make_unique([df.columns[0] for df in cols])):
            df.columns = [name]
        merg = lambda x, y: x.merge(y, how='outer', right_index=True,
                                    left_index=True)
        info = reduce(merg, cols)
        info.columns = summ.columns
        info = info.fillna('')
    
        if show is 't':
            note = ['\t t statistics in parentheses.']
        if show is 'se':
            note = ['\t Std. error in parentheses.']
        if show is 'p':
            note = ['\t pvalues in parentheses.']
        if stars:
            note +=  ['\t * p<.1, ** p<.05, ***p<.01']
    
    #Here  I tried two ways to put extra text in index-location or
    # columns-location,finally found the former is better.
    
    
        note_df = pd.DataFrame([ ],index=['note:']+note,
                                                    columns=summ.columns).fillna('')
        
        if title is not None:
            title = str(title)
        else:
            title = '\t Results Summary'
        
        # Here I tried to construct a title DataFrame and 
        # adjust the location of title corresponding to the length of columns. 
        # But I failed because of not good printing effect.
        
        title_df = pd.DataFrame([],index=[title],columns=summ.columns).fillna('')
        
        smry = Summary()
        smry.add_df(title_df,header=False,align='l') # title DF
        smry.add_df(summ, header=True, align='l') # params DF
        smry.add_df(info, header=False, align='l') # model information DF
        smry.add_df(note_df, header=False, align='l') # extra text DF
        return smry