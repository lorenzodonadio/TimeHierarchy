import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
from .BaseFuncs import BaseFuncs

class ForecastETS(BaseFuncs):
    '''Empty docstring'''
    def __init__(self,
                 col_name,
                 n_nodes,
                 hierarchy = None,
                 hierarchy_short = None,
                 use_short = True):
        super().__init__()
        '''Empty docstring'''
        if use_short and hierarchy_short is None:
            raise ValueError('Must provide hierarchy_short')
        if not use_short and hierarchy is None:
            raise ValueError('Must provide hierarchy')   
            
        self.col_name = col_name
        self.n_nodes = n_nodes
        self.use_short = use_short
        self.hierarchy = hierarchy
        self.hierarchy_short = hierarchy_short
        
    def fit(self,trend = 'add',
                seasonal = 'add',
                seasonal_periods = 100,
                disable_tqdm = True,
                damped_trend = True):

        if type(seasonal_periods) == type(list()):
            assert len(seasonal_periods) == len(self.n_nodes)
        elif type(seasonal_periods) == type(int()):
            seasonal_periods = len(self.n_nodes)*[seasonal_periods]
        else:
            raise ValueError('Wrong input for seasonal_periods, either int or list of ints of lenght equal to tree levels')
        '''Empty docstring'''
        if self.use_short:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_tree = {}

                for k in tqdm(self.hierarchy_short, disable = disable_tqdm):
                    tmp_dict = {}

                    for k2 in self.hierarchy_short[k]:
                        tmp = self.hierarchy_short[k][k2] 
                        tmp_dict[k2] = ExponentialSmoothing(tmp[self.col_name],
                                                            trend=trend,
                                                            seasonal=seasonal,
                                                            seasonal_periods=seasonal_periods[k],
                                                            damped_trend = damped_trend).fit()

                    self.model_tree[k] =  tmp_dict
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_tree = {}
                for k in tqdm(self.hierarchy, disable = disable_tqdm):
                    self.model_tree[k] =  ExponentialSmoothing(self.hierarchy[k][self.col_name],
                                                            trend=trend,
                                                            seasonal=seasonal,
                                                            seasonal_periods=seasonal_periods[k],
                                                            damped_trend = damped_trend).fit()
                
        self._compute_errors()

    
    def forecast(self,h):
        '''Empty docstring'''
        self.forecast_tree = {}
        if self.use_short:
            for k in self.model_tree:
                tmp_dict = {}
                for k2 in self.model_tree[k]:

                    tmp_dict[k2] = self.model_tree[k][k2].forecast(h)
                self.forecast_tree[k] =  tmp_dict
        else:
            for k,node in zip(self.model_tree,self.n_nodes):
                tmp_dict = {}
                tmp = self.model_tree[k].forecast(int(h*node))
                for k2 in range(int(node)):
                    tmp_dict[k2] = tmp[k2::int(node)]
                self.forecast_tree[k] = tmp_dict

        self.yhat = self.compute_y_hat(self.forecast_tree)