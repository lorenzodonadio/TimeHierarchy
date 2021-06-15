import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from .BaseFuncs import BaseFuncs

class ForecastARIMA(BaseFuncs):
    """Empty docstring"""
    def __init__(self,
                 col_name,
                 n_nodes,
                 hierarchy = None,
                 hierarchy_short = None,
                 use_short = True):
        '''Empty docstring'''
        super().__init__()
        if use_short and hierarchy_short is None:
            raise ValueError('Must provide hierarchy_short')
        if not use_short and hierarchy is None:
            raise ValueError('Must provide hierarchy')   
            
        self.col_name = col_name
        self.n_nodes = n_nodes
        self.use_short = use_short
        self.hierarchy = hierarchy
        self.hierarchy_short = hierarchy_short
        
    def fit(self,order = (1,0,0),
            seasonal_order = (0,0,0,0),
            disable_tqdm = True):
        '''
         We would fit the same arima model to each level of the hierarchy, when we pass a tuple...
         so order and seasonal order can be a list of tuples of length(k)
         
         '''
        if type(order) is list and type(seasonal_order) is list:
            assert len(order) == len(seasonal_order) == len(self.n_nodes)
        elif type(order) is tuple and type(seasonal_order) is tuple:
            order = len(self.n_nodes)*[order]
            seasonal_order = len(self.n_nodes)*[seasonal_order]

        if self.use_short:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_tree = {}

                for k in tqdm(self.hierarchy_short, disable = disable_tqdm):
                    tmp_dict = {}

                    for k2 in self.hierarchy_short[k]:
                        tmp_dict[k2] = SARIMAX(self.hierarchy_short[k][k2][self.col_name].values,
                                            order = order[k],
                                            seasonal_order = seasonal_order[k],
                                            initialization='approximate_diffuse').fit()
                        tmp_dict[k2].fittedvalues = pd.Series(tmp_dict[k2].predict())
                        
                    self.model_tree[k] =  tmp_dict 
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_tree = {}
                
                for k in tqdm(self.hierarchy, disable = disable_tqdm):    
                    self.model_tree[k] = SARIMAX(self.hierarchy[k][self.col_name].values,
                                            order = order[k],
                                            seasonal_order = seasonal_order[k],
                                            initialization='approximate_diffuse').fit()
                    self.model_tree[k].fittedvalues = pd.Series(self.model_tree[k].predict())
        self._compute_errors()
                
    def forecast(self,h):
        '''Empty docstring'''
        self.forecast_tree = {}
        if self.use_short:
            for k in self.model_tree:
                tmp_dict = {}
                for k2 in self.model_tree[k]:
                ###DO YOUR THING WITH ARIMA
                    tmp_dict[k2] = pd.Series(self.model_tree[k][k2].forecast(h))
                self.forecast_tree[k] =  tmp_dict
        else:
            for k,node in zip(self.model_tree,self.n_nodes):
                tmp_dict = {}
                ###DO YOUR THING WITH ARIMA
                tmp = pd.Series(self.model_tree[k].forecast(int(h*node)))
                for k2 in range(int(node)):
                    tmp_dict[k2] = tmp[k2::int(node)]
                self.forecast_tree[k] = tmp_dict

        self.yhat = self.compute_y_hat(self.forecast_tree)