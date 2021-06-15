import numpy as np
import pandas as pd
import warnings
import pdb
from statsmodels.tsa.stattools import acf

from scipy.linalg import block_diag

from sklearn.covariance import GraphicalLasso ##,EmpiricalCovariance,OAS
from ..utils.regressors import RegSVR,RegLGBM
from ..utils.utils import create_sum_mat, _to_block_diag,compute_y_hat

class Reconciliation():
    '''Empty docstring'''
    def __init__(self,
                n_nodes,
                col_name,
                method = 'OLS',
                error_df = None,
                model_tree = None,
                hierarchy_short = None, 
                **kwargs):
        """ Init docstring:
        
        Possibel methods:
        GLS_methods : 'BU','COV','blockCOV','blockGLASSO','GLASSO','VAR','STR','OLS','CCS','markov'
        ML_methods = 'LGBM', 'SVM """
        self.col_name = col_name
        self.n_nodes = n_nodes
        self.levels = [i for i in range(len(self.n_nodes))] # hierarchy levels
        self.sum_mat = create_sum_mat(n_nodes = self.n_nodes)
        self.method = method

        self._GLS_methods =  ['BU','COV','blockCOV','blockGLASSO','GLASSO','VAR','STR','OLS','CCS','markov']
        self._ML_methods = ['LGBM','SVM']
        
        #self.error_df = error_df  ##no  real need to store these variables???
        #self.model_tree = model_tree
        
        if method in self._GLS_methods:
            self.inv = self.compute_inv(error_df = error_df,**kwargs)
            self.G = self.compute_G()
        elif method in self._ML_methods:
            self.ml_recon_model = self.train_ml_recon(model_tree = model_tree,
                                                      hierarchy_short = hierarchy_short,
                                                      method = self.method,
                                                      **kwargs)
        else:
            raise ValueError('Invalid Reconciliation method')
    
    def _create_markov(self,error_df):
        '''Create the markov scaling block diagonal matrix'''
        idx_list = [0]+list(np.cumsum(self.n_nodes))
        rho = []
        for i,j in zip(idx_list[:-1],idx_list[1:]):
            tmp = np.squeeze(error_df.iloc[:,i:j].values.reshape(-1,1))
            rho.append(acf(tmp,nlags=1,fft=True)[1]) # acf -> statsmodels.tsa.stattools.acf
        blocks = []
        for k,i in enumerate(self.n_nodes):
            tmp = np.eye(i)
            for j in range(i):
                for w in range(i):
                    if j!=w:
                        tmp[j,w] = rho[k]**(abs(j-w))
            blocks.append(tmp)

        return block_diag(*blocks)
    
    
    def compute_inv(self,error_df = None,method = None,to_remove = 0.2,alpha = 0.5,lda = 0.5,**kwargs):
        """Estimates the inverse of the Error Covariance matrix
        
        parameters:
        error_df: pd.DataFrame containing the errors 
        
        to_remove (int):number of rows to remove from the error_df prior to the covariance estimation
        must be less than 80% of the df length
        alpha: (0-1) only affects GLASSO
        lda (lambda)(0-1) only affects CCS (cross correlation shrinkage) 
        kwargs: convmethod: str, either transpose or numpy: how to calculate the full covariance matrix"""
        method = self.method if method is None else method
        covmethod = kwargs['covmethod'] if  'covmethod' in kwargs else 'transpose'
        assert 0 < to_remove<0.9*len(error_df) 
        if 0 < to_remove < 1:
            to_remove = int(to_remove*len(error_df))
        
        error_df = error_df[to_remove:]
        
        if method == 'COV': # full covariance weighted least squares
            if covmethod == 'numpy':
                w = np.cov(error_df.T.values)
                return np.linalg.pinv(w) ## WLSV
            elif covmethod == 'transpose':
                w = np.matmul(error_df.values.T,error_df.values)/(error_df.shape[0]-1)
                #pdb.set_trace()
                return np.linalg.pinv(w)
            else:
                raise ValueError('Incorrect covmethod: possible numpy or transpose')
        elif method == 'blockCOV': # block covariance weighted least squares a.k.a autocovariance scaling
            if covmethod == 'numpy':
                w = _to_block_diag(np.cov(error_df.T.values),self.n_nodes)
                return np.linalg.pinv(w) ## WLSV
            elif covmethod == 'transpose':
                w = np.matmul(error_df.values.T,error_df.values)/(error_df.shape[0]-1)
                return np.linalg.pinv(_to_block_diag(w,self.n_nodes))
            else:
                raise ValueError('Incorrect covmethod: possible numpy or transpose')
            
        elif method == 'GLASSO': # glasso covariance weighted least squares
            return GraphicalLasso(alpha = 0.5,max_iter = 400,mode = 'cd').fit(error_df.values).precision_
            
        elif method == 'blockGLASSO': # block glasso covariance weighted least squares
            return _to_block_diag(GraphicalLasso(alpha = 0.5,
                            max_iter = 400,mode = 'cd').fit(error_df.values).precision_,self.n_nodes)  
            
        elif method == 'VAR': # variance weighted least squares
            w = np.diag(np.var(error_df.T.values,axis = 1))
            return np.linalg.pinv(w)
            
        elif method == 'STR': # structurally weighted least squares
            w = np.diag(np.sum(self.sum_mat,axis = 1))
            return np.linalg.pinv(w) ## WLSS
            
        elif method == 'OLS': # ordinary least squares
            return np.eye(self.sum_mat.shape[0])
            
        elif method == 'BU': # bottom up
            return None
        
        elif method == 'CCS': #cross correlation shrinkage
            R = np.corrcoef(error_df.T.values)
            hvar_12 = np.diag(np.sqrt(np.var(error_df.T.values,axis = 1)))
            Rshrink = (1-lda)*R + lda*np.eye(R.shape[0])
            w = np.matmul(hvar_12,np.matmul(Rshrink,hvar_12))
            return np.linalg.pinv(w)
            
        elif method == 'markov':
            hvar_12 = np.diag(np.sqrt(np.var(error_df.T.values,axis = 1)))
            w = np.matmul(hvar_12,np.matmul(self._create_markov(error_df),hvar_12))
            return np.linalg.pinv(w)
        
    def train_ml_recon(self,model_tree,hierarchy_short,method='LGBM',
                            to_remove = 0, #number of initial samplees to skip
                            weight_hierarchy = True,   
                            **kwargs):
        if weight_hierarchy: # weight X according to the Number of child nodes (only bottom level children)
            self.w = [i*[j] for i,j in zip(self.n_nodes,[int(max(self.n_nodes)/(i)) for i in self.n_nodes])]
            self.w = [item for sublist in self.w for item in sublist]
        else:
            self.w = 1
        # Y is always the same, whether we 'use_short' or nor
        y = [] # create X (all nodes from the tree)
        k = self.levels[-1]
        for k2 in range(self.n_nodes[k]):
            y.append(hierarchy_short[k][k2][self.col_name].values.reshape(-1,1))
        y = np.hstack(y)
        
        x=[]
        if type(model_tree[0]) == type(dict()): # short format
            
            for k in self.levels:
                for k2 in hierarchy_short[k]:
                    # attribute .fittedvalues  is a must-> enfore it in ARIMA TBATS and LGBM -> DONE
                    x.append(model_tree[k][k2].fittedvalues.values.reshape(-1,1))

        else:                     # long format
            
            for k in self.levels:
                x.append(model_tree[k].fittedvalues.values.reshape(-1,self.n_nodes[k]))
                
        x = np.hstack(x)/self.w
        ## Remove the first observations 
        assert 0 < to_remove < 0.9*x.shape[0]
        if 0 < to_remove < 1:
            to_remove = int(to_remove*x.shape[0])
                
        if method == 'LGBM':
            return  RegLGBM(x[to_remove:],y[to_remove:],**kwargs) # actually train the model
        elif method == 'SVM':
            #pdb.set_trace()
            return  RegSVR(x[to_remove:],y[to_remove:],**kwargs) # actually train the model
        else:
            raise ValueError('Wrong ML-Reconciliation Method->only LGBM , SVM')
        ## return x,y ##only for debug
        
    def compute_G(self,inv = None,method = None):
        inv = self.inv if inv is None else inv
        method = self.method if method is None else method
        if self.method == 'BU': # bottom up
            G = np.zeros((max(self.n_nodes),sum(self.n_nodes)))
            G[:,-1] = 1
            return G
        else:
            transpose = self.sum_mat.T
            xaux = np.matmul(np.matmul(transpose,inv),self.sum_mat)
            return np.matmul(np.matmul(np.linalg.inv(xaux),transpose),inv)
        
    def gls_recon(self,yhat, G = None):
        '''Empty docstring'''
        G = self.G if G is None else G
        return np.matmul(np.matmul(self.sum_mat,G),yhat)
        
    def ml_recon(self,forecast_tree):
        '''performs ml reconciliation, steps: 
        1. from forecast_tree (dict(dict(pd.Series))) create and array or predictors (xnew)
        2. use the pretrained model (by train_ml_recon) self.ml_recon_model to get bottom level predictions
        3. aggregate bottom level using BU method and return'''
        xnew = []  # predict reconciled forecastsreconciled
        for k in self.levels:
            for k2 in range(self.n_nodes[k]):
                xnew.append(forecast_tree[k][k2].values.reshape(-1,1))
        xnew = np.hstack(xnew)/self.w
        #bottom level predictions
        ypred = np.array([self.ml_recon_model.predict(xnew[i,:]) for i in range(xnew.shape[0])]) 
        return np.matmul(self.sum_mat,ypred.T) # BU method
    
    def reconcile(self,forecast_tree):
        '''wrapper around gls_recon and ml_recon, so only one function is called for all the supported methods
        gls_recon takes as input yhat and not forecast tree so we need to create it'''
        if self.method in self._ML_methods:
            return self.ml_recon(forecast_tree)
        elif self.method in self._GLS_methods:
            yhat = compute_y_hat(forecast_tree)
            return self.gls_recon(yhat)