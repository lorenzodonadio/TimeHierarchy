import numpy as np
import pandas as pd

def create_sum_mat(k=None,n_nodes = None):
    '''Empty docstring'''
    k = [int(max(n_nodes)/(i)) for i in n_nodes] if k is None else k
    k,m = np.sort(k)[::-1] , max(k)
    return np.vstack([np.kron(np.eye(int(m/kk)),np.ones((1,kk))) for kk in k])

def split_weekends(df,time_name ='timestamp'):
        '''Empty docstring'''
        weekday = df[time_name].apply(lambda x: x.weekday())  # remove saturday and sunday
        return df[[i not in [5,6] for i in weekday]],df[[i  in [5,6] for i in weekday]]
    
def _to_block_diag(x,n_nodes):
    '''enforces matrix x to be a block diagonal matrix,
    accoridng to n_nodes'''
    for n,i in enumerate(np.cumsum(n_nodes)[:-1]):
        for j in range(int(n_nodes[n])-1,int(i)):
            x[j,int(i):] = 0
            x[int(i):,j] = 0
    return x

def compute_y_hat(forecast_tree):  #try to keep it the same, yep is the same...
    '''Empty docstring'''
    l = len(forecast_tree[0][0])
    yhat = [[] for i in range(l)]
    for k in forecast_tree:
        for k2 in forecast_tree[k]:
            for i in range(l):
                yhat[i].append(forecast_tree[k][k2].values[i])

    return  np.array(yhat).T

def _return_time_feats(series,feat):
        """
        subscript _x or _y indicate mapping with cos and sin respectivly
        series : pd.Series of dtype pd.datetime
        returns hour_x, hour_y, month_x, month_y, doy_x, doy_y (doy = dayofyear) columns
        if requested by feats list default is which one is contained in feats
        """

        if 'hour_x' in feat:
            return np.cos(2*np.pi*series.apply(lambda x: x.hour).values/23)    
        elif 'hour_y' in feat:
            return np.sin(2*np.pi*series.apply(lambda x: x.hour).values/23)

        elif 'month_x' in feat:
            return np.cos(2*np.pi*series.apply(lambda x: x.month).values/11) 
        elif 'month_y' in feat:
            return np.sin(2*np.pi*series.apply(lambda x: x.month).values/11)
        
        elif 'doy_x' in  feat:
            return np.cos(2*np.pi*series.dt.dayofyear/364)
        elif 'doy_y' in  feat:
            return np.sin(2*np.pi*series.dt.dayofyear/364)
        
        elif 'week_x' in  feat:
            return np.cos(2*np.pi*series.dt.dayofweek/6)
        elif 'week_y' in  feat:
            return np.sin(2*np.pi*series.dt.dayofweek/6)
        else:
            raise ValueError('Incorrect Feature-> only: hour_x, hour_y, month_x, month_y, doy_x, doy_y')