import numpy as np
import pandas as pd
import warnings

from .forecast_models.ETS import ForecastETS
from .forecast_models.ARIMA import ForecastARIMA
from .forecast_models.TBATS import ForecastTBATS
from .forecast_models.LGBM import ForecastLGBM

from .utils.utils import split_weekends,compute_y_hat,_return_time_feats
from .recon.recon import Reconciliation

import pdb
class TimeHierarchy():
    '''Empty docstring'''
    def __init__(self,df,n_nodes,col_name,
                 time_name ='timestamp',
                 sep_weekend = True,
                 test_size = None,
                 verbose = 0,
                 features = [],
                 time_features = []):
        '''Empty docstring'''
        max_node,min_node = max(n_nodes),min(n_nodes)
        self.verbose = verbose
        self.test_size = test_size
        self.n_nodes = [int(i/min_node) for i in np.sort(n_nodes)]
        ## make sure that n_nodes provides a coherent tree structure
        assert all(max(self.n_nodes) % n == 0 for n in self.n_nodes) 
        self.k = [int(max_node/(i*min_node)) for i in self.n_nodes]
        
        self.time_name = time_name
        self.col_name = col_name
        self.features = features
        self.time_features = time_features
        #
        # if n_nodes specifies it we aggregate data before tree creation, 
        # this is done to reduce the time-step in the bottom level of the tree
        if min_node>1:
            df2 = pd.DataFrame(df[self.time_name][::min_node]).reset_index(drop=True)
            df2[self.col_name] = df[self.col_name].groupby(df.index // min_node).sum()
            for f in features:
                df2[f] = df[f].groupby(df.index // min_node).mean()
            
        else:
            df2 = df.copy()
 
        if sep_weekend:
            self.df,self.df_end = split_weekends(df2)
            self.df,self.df_end = self.df.reset_index(drop=True),self.df_end.reset_index(drop=True)
        else:
            df2 = df2[:max(k)*(df2.shape[0]//max(k))]
            self.df,self.df_end = df2,pd.DataFrame()
            
        self.create_hierarchy(self.df,self.col_name) # create both short and long format hierarchies
        self.y_true_test = compute_y_hat(self.hierarchy_short_test)[1::][0]
        
        self.timedeltas = [] # time deltas between values for the hierarchy levels
        for k in self.hierarchy:
            self.timedeltas.append(self.hierarchy[k][self.time_name][1]-self.hierarchy[k][self.time_name][0])
    
    def __repr__(self):
        s = ''
        for k in self.hierarchy:
            s +='K = '+str(k) + ' --> nodes = '
            for i in np.unique(self.hierarchy[k]['group_id']):
                s += str(int(i))+' '
            s +='\n'
        return s
    
    def create_hierarchy(self,df,col_name,k = None,time_name = None):
        '''Empty docstring'''
        k = k if k is not None else self.k
        time_name = time_name if time_name is not None else self.time_name
        df = df[:int(max(k)*(df.shape[0]//max(k)))].copy()
        
        self.hierarchy = {} # Dict[int,pd.Dataframe]
        for j,kk in enumerate(k):
            self.hierarchy[j] = pd.DataFrame(df[time_name][::kk].values,columns=[time_name])
            if self.verbose:
                print(len(df)//kk)
                
            _aux = np.arange(len(df))//kk
            self.hierarchy[j][col_name] = df[col_name].groupby(_aux).sum()
            for f in self.features:
                self.hierarchy[j][f] = df[f].groupby(_aux).mean()
                     #create time features 
            for f in self.time_features:
                self.hierarchy[j][f] = _return_time_feats(self.hierarchy[j][time_name],f)   
                
        len_max = len(self.hierarchy[0])
        # create short hierarchy n values per node
        self.hierarchy_short = {} # Dict[int,Dict[int,pd.Dataframe]]
        for j in range(len(k)):
            self.hierarchy[j]['group_id'] = np.tile(np.arange(k[0]/k[j]),len_max)
            tmp = {}
            for i in np.unique(self.hierarchy[j]['group_id']):
                tmp[i] = self.hierarchy[j][self.hierarchy[j]['group_id'] == i].reset_index(drop=True)
                
                for f in self.features:
                    tmp[i][f] = self.hierarchy[0][f]
                
            self.hierarchy_short[j] = tmp
            
        if self.test_size is not None: ## create test trees, in both formats
            self.hierarchy_short_test = {}
            self.hierarchy_test = {}
            for k,n in zip(self.hierarchy,self.n_nodes):
                self.hierarchy_test[k] = self.hierarchy[k][-self.test_size*int(n):]  #long format test treee
                self.hierarchy[k] = self.hierarchy[k][:-self.test_size*int(n)]
                tmp_dict = {}
                for k2 in self.hierarchy_short[k]:  #short format test tree
                    tmp_dict[k2] = self.hierarchy_short[k][k2][-self.test_size:]
                    self.hierarchy_short[k][k2] = self.hierarchy_short[k][k2][:-self.test_size]
                self.hierarchy_short_test[k] = tmp_dict
        self.len_max = len(self.hierarchy[0])
        
                
    
    def fit_ets(self,use_short=True,trend = 'add',seasonal = 'add',seasonal_periods = 100,**kwargs):
        '''Empty docstring'''
        self.ETS = ForecastETS(self.col_name,self.n_nodes,
                   hierarchy = self.hierarchy,
                   hierarchy_short = self.hierarchy_short,
                   use_short = use_short)
        
        self.ETS.fit(trend = trend,seasonal = seasonal,seasonal_periods = seasonal_periods,**kwargs)
        
    def fit_arima(self,use_short=True,order = (1,0,0),seasonal_order = (0,0,0,0),**kwargs):
        '''Empty docstring'''
        self.ARIMA = ForecastARIMA(self.col_name,self.n_nodes,
                     hierarchy = self.hierarchy,
                     hierarchy_short = self.hierarchy_short,
                     use_short = use_short)
        
        self.ARIMA.fit(order = order, seasonal_order = seasonal_order,**kwargs)
        
    def fit_tbats(self,use_short=True,seasonal_periods = None,**kwargs):
        '''Empty docstring'''
        self.TBATS = ForecastTBATS(self.col_name,self.n_nodes,
                     hierarchy = self.hierarchy,
                     hierarchy_short = self.hierarchy_short,
                     use_short = use_short)
        
        self.TBATS.fit(seasonal_periods = seasonal_periods,**kwargs)
        
    def fit_lgbm(self,use_short=True,look_back = 24,features=None,**kwargs):
        '''Empty docstring'''
        
        self.LGBM = ForecastLGBM(self.col_name,self.n_nodes,
                    hierarchy = self.hierarchy,
                    hierarchy_short = self.hierarchy_short,
                    use_short = use_short,
                    look_back = look_back,
                    features = features)
        
        self.LGBM.fit(**kwargs)
        
    def _get_forecast_model(self,forecast_method):
        assert forecast_method in dir(self)
        
        if forecast_method == 'ETS':
            return self.ETS
        elif forecast_method == 'LGBM':
            return self.LGBM
        elif forecast_method == 'ARIMA':
            return self.ARIMA
        elif forecast_method == 'TBATS':
            return self.TBATS
    
    def create_recon(self,
                     recon_method,
                     forecast_method,
                     **kwargs):
        '''Wrapper Around Reconciliation class
        initialized the class and adds it as an attribute(.Recon) to the Time_Hierarchy class
        
        Possibel recon_method:
        GLS_methods : 'BU','COV','blockCOV','blockGLASSO','GLASSO','VAR','STR','OLS','CCS','markov'
        ML_methods = 'LGBM'
        
        forecast_method (must be already trained) : 'ETS','LGBM','TBATS','ARIMA'        
        **kwargs are given to the Reconciliation class'''
        forecast_model = self._get_forecast_model(forecast_method)
        
        self.Recon = Reconciliation(self.n_nodes,self.col_name,
                               method = recon_method,
                               error_df = forecast_model.error_df,
                               model_tree = forecast_model.model_tree,
                               hierarchy_short = self.hierarchy_short,
                               **kwargs)
