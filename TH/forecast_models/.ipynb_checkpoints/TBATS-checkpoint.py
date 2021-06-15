import numpy as np
import pandas as pd
import warnings
from tbats import TBATS
from tqdm import tqdm
from .BaseFuncs import BaseFuncs

class ForecastTBATS(BaseFuncs):
    """Wrapper around the TBATS class form: https://pypi.org/project/tbats/ 
    so that it can be fitted to a tree hierarchy"""
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
        
    def fit(self,use_box_cox=False,
            box_cox_bounds=(0, 1),
            use_trend=False,
            use_damped_trend=None,
            seasonal_periods=None,
            use_arma_errors=True,
            show_warnings=True,
            n_jobs=None,
            multiprocessing_start_method='spawn',
            context=None,
            disable_tqdm = True):
        '''
         We would fit the TBATS model to a tree_hierarchy, seasonal_periods is the most important parameter,
         it can either be None, a list of lists of length n_nodes (a different seasonal order for each level)
         or a simple list containing the seasonal periods for all levels.
         
         '''
        
        if seasonal_periods is None:
            seasonal_periods = len(self.n_nodes)*[None]
        elif type(seasonal_periods) is list:
            if any(type(l) is list for l in seasonal_periods): # different seasonal_periods for each level
                pass
            else:
                seasonal_periods = len(self.n_nodes)*[seasonal_periods]
            assert len(seasonal_periods) == len(self.n_nodes) 
        else:
            raise ValueError('Incorrect seasonal_periods parameter')

        if self.use_short:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_tree = {}

                for k in tqdm(self.hierarchy_short, disable = disable_tqdm):
                    tmp_dict = {}

                    for k2 in self.hierarchy_short[k]:
                        tmp_dict[k2] = TBATS(use_box_cox=use_box_cox,
                                            box_cox_bounds=box_cox_bounds,
                                            use_trend=use_trend,
                                            use_damped_trend=use_damped_trend,
                                            seasonal_periods=seasonal_periods[k],
                                            use_arma_errors=use_arma_errors,
                                            show_warnings=show_warnings,
                                            n_jobs=n_jobs,
                                            multiprocessing_start_method=multiprocessing_start_method,
                                            context=context).fit(self.hierarchy_short[k][k2][self.col_name].values)
                        
                        tmp_dict[k2].fittedvalues = pd.Series(tmp_dict[k2].y_hat)

                    self.model_tree[k] =  tmp_dict 
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_tree = {}
                
                for k in tqdm(self.hierarchy, disable = disable_tqdm):    
                    self.model_tree[k] = TBATS(use_box_cox=use_box_cox,
                                            box_cox_bounds=box_cox_bounds,
                                            use_trend=use_trend,
                                            use_damped_trend=use_damped_trend,
                                            seasonal_periods=seasonal_periods[k],
                                            use_arma_errors=use_arma_errors,
                                            show_warnings=show_warnings,
                                            n_jobs=n_jobs,
                                            multiprocessing_start_method=multiprocessing_start_method,
                                            context=context).fit(self.hierarchy[k][self.col_name].values)
                    
                    self.model_tree[k].fittedvalues = pd.Series(self.model_tree[k].y_hat)
                    
        self._compute_errors()
                
    def forecast(self,h):
        '''Empty docstring'''
        self.forecast_tree = {}
        if self.use_short:
            for k in self.model_tree:
                tmp_dict = {}
                for k2 in self.model_tree[k]:
                ###DO YOUR THING WITH TBATS
                    tmp_dict[k2] = pd.Series(self.model_tree[k][k2].forecast(h))
                self.forecast_tree[k] =  tmp_dict
        else:
            for k,node in zip(self.model_tree,self.n_nodes):
                tmp_dict = {}
                ###DO YOUR THING WITH TBATS
                tmp = pd.Series(self.model_tree[k].forecast(int(h*node)))
                for k2 in range(int(node)):
                    tmp_dict[k2] = tmp[k2::int(node)]
                self.forecast_tree[k] = tmp_dict

        self.yhat = self.compute_y_hat(self.forecast_tree)