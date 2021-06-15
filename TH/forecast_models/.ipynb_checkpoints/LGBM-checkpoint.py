import numpy as np
import pandas as pd
import warnings
from tbats import TBATS
from tqdm import tqdm
from .BaseFuncs import BaseFuncs
from ..utils.regressors import RegLGBM
from lightgbm import LGBMRegressor

import pdb

class ForecastLGBM(BaseFuncs):
    def __init__(self,
                 col_name,
                 n_nodes,
                 hierarchy = None,
                 hierarchy_short = None,
                 use_short = True,
                 look_back = 96,
                 features = None):
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
        self.look_back = look_back
        self.features = features
        
    def prepare_y(self,df,horizon = 1):
        y = []
        for i in range(0,df.shape[0]-(self.look_back + horizon)):
            tmp_y = df[self.col_name][i+self.look_back : i+self.look_back+horizon].values.reshape(1,-1).ravel()
            y.append(tmp_y)

        return np.array(y)

    def prepare_x(self, df,horizon = 1,features = None):
        df = self.df if df is None else df
        x = []
        for i in range(0,df.shape[0]-(self.look_back + horizon)):
            
            tmp_x = df[self.col_name][i:i+self.look_back].values.reshape(1,-1).ravel()
            
            if features is None:
                x.append(tmp_x)
            elif type(features) is type(list()):
                ex = df[features].iloc[i+self.look_back + horizon].values #exhugenous features
                x.append(np.hstack([tmp_x,ex]))
            else:
                raise ValueError('features must be a list of column names contained in df')
            #pdb.set_trace()
        return np.array(x)
    
    def fit(self,eval_metric = 'l2',
                sample_weight = None,
                disable_tqdm = True,
                **kwargs):
            self.model_tree = {}

            if self.use_short:
                for k in tqdm(self.hierarchy_short, disable = disable_tqdm): # train the models
                    tmp_dict = {}
                    for k2 in self.hierarchy_short[k]:
                        X = self.prepare_x(self.hierarchy_short[k][k2],features = self.features)
                        y = self.prepare_y(self.hierarchy_short[k][k2])
                        tmp_dict[k2] = LGBMRegressor(**kwargs).fit(X,
                                                           y.ravel(),
                                                           eval_metric = eval_metric,
                                                           sample_weight = sample_weight)
                    self.model_tree[k] = tmp_dict
                    # create in sample prediction for the reconciliation errors
                    for k2 in self.hierarchy_short[k]:
                        ys = [] # ys = y sample ->in sample predict
                        series = self.hierarchy_short[k][k2][self.col_name]
                        for i in range(len(series)-self.look_back-1):
                            tmp = series[i:i+self.look_back] #endogenous features
                            if self.features is not None:
                                ex = self.hierarchy_short[k][k2][self.features].iloc[i+self.look_back+1].values
                                ex  = ex if np.size(ex) else None # exogenous features
                            else:
                                ex = None
                            ys.append(self._one_step_ahead(tmp,self.model_tree[k][k2],ex = ex)[0])
                        
                        fitted_values = self.look_back*[ys[0]] + ys
                        ld = len(self.hierarchy_short[k][k2]) - len(fitted_values)
                        # add extra data for point where we could not forecast due to exogenous regressors
                        self.model_tree[k][k2].fittedvalues = pd.Series(np.array(fitted_values+ld*[np.mean(ys)]))
            else:

                for k in tqdm(self.hierarchy, disable = disable_tqdm):
                    horizon = self.n_nodes[k]
                    ys = [] # ys = y sample ->in sample predict
                    #fit the model
                    X = self.prepare_x(self.hierarchy[k],horizon = horizon,features = self.features)
                    y = self.prepare_y(self.hierarchy[k],horizon = horizon)
                    self.model_tree[k] = RegLGBM(X,y,
                                                  eval_metric = eval_metric,
                                                  sample_weight = sample_weight,
                                                  **kwargs)

                    # create in sample prediction for the reconciliation errors
                    series  = self.hierarchy[k][self.col_name]
                    node = self.n_nodes[k]
                    for i in range(int(np.ceil((len(series)-self.look_back)/node))-horizon):
                        if self.features is not None:
                            ex = self.hierarchy[k][self.features].iloc[i+self.look_back+horizon].values
                            ex = ex if np.size(ex) else None # ex ogenous features
                        else:
                            ex = None
                        tmp = series[i*node:i*node+self.look_back] #endogenous features
                        ys.extend(self._one_step_ahead(tmp,self.model_tree[k],ex = ex))

                    # make the lengths consistent, the redundant data will be removed after
                    fitted_values = self.look_back*[ys[0]] + ys
                    ld = len(self.hierarchy_short[k][0]) - len(fitted_values)
                    # add extra data for point where we could not forecast due to exogenous regressors
                    fitted_values = np.array(fitted_values+ld*[np.mean(ys)]) 
                    self.model_tree[k].fittedvalues = pd.Series(fitted_values[:len(series)]) 
            # compute in sample errors for reconcilitation         
            self._compute_errors(to_remove = self.look_back)
            
            if self.error_df.isna().any().any():
                self.error_df = self.error_df.fillna(method='ffill')
            
    def _one_step_ahead(self,series,lgbm_model,ex=None):
        if ex is None:
            return lgbm_model.predict(np.array(series[-self.look_back:].values).reshape(1, -1)) 
        else:
            x_in = np.hstack([np.array(series[-self.look_back:].values),ex]).reshape(1, -1)
            return lgbm_model.predict(x_in)
            
    def forecast(self,h,ex=None):
        '''Empty docstring'''
        self.forecast_tree = {}
        if self.use_short:
            for k in self.model_tree:
                tmp_dict = {}
                for k2 in self.model_tree[k]:
                ###DO YOUR THING WITH LGBM
                    series = self.hierarchy_short[k][k2][self.col_name].copy()
                    for i in range(h):
                        if ex is None:
                            y_pred = self._one_step_ahead(series,self.model_tree[k][k2])[0]
                        else:
                            #pdb.set_trace()
                            y_pred = self._one_step_ahead(series,self.model_tree[k][k2],ex = ex[i,:])[0] # forecasted value
                        series = series.append(pd.Series(data = y_pred,index = [series.index.argmax() + 1]))
                    tmp_dict[k2] = series[-h:]
                self.forecast_tree[k] =  tmp_dict
        else:

            for k,node in zip(self.model_tree,self.n_nodes):

                mdl = self.model_tree[k]
                series = self.hierarchy[k][self.col_name]
                for i in range(h):
                    if ex is None:
                        series = series.append(pd.Series(self._one_step_ahead(series,mdl))).reset_index(drop=True)
                    else:
                        series = series.append(pd.Series(self._one_step_ahead(series,mdl,ex=ex[i,:]))).reset_index(drop=True)
 
                tmp = series[-int(h*self.n_nodes[k]):] # retrieve the forecasted values
                tmp_dict = {}
                for k2 in range(int(node)):
                    tmp_dict[k2] = tmp[k2::int(node)]
                self.forecast_tree[k] = tmp_dict

        self.yhat = self.compute_y_hat(self.forecast_tree)