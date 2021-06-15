import numpy as np
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
#import pdb
## Add ANN ??
class RegSVR():
    def __init__(self,x,y, 
                 sample_weight = None,
                 verbose = False,
                 C=100,
                 epsilon=0.05,
                 gamma='auto',
                 **kwargs):
        """Empty docstring"""
        self.y_size = y.shape[1]
        self._models_list = []

        for i in range(self.y_size):
            if verbose:
                print(i,end = ' ')
            lgbm = SVR(C = C,epsilon = epsilon,gamma = gamma)
            self._models_list.append(
                    lgbm.fit(X = x, y = y[:,i],
                    sample_weight=sample_weight))

    def predict(self, x):
        """Empty docstring"""
        y_out = []
        for i in range(self.y_size):
            #pdb.set_trace()
            y_out.append(self._models_list[i].predict(np.array(x).reshape(1,-1))[0])
        return np.array(y_out)
    
class RegLGBM():
    def __init__(self,x,y, 
                 sample_weight = None,
                 verbose = False,
                 max_depth = -1,
                 num_leaves = 31,
                 learning_rate=0.1,
                 eval_metric = 'l2',
                 **kwargs):
        """Empty docstring"""
        self.y_size = y.shape[1]
        self._models_list = []

        for i in range(self.y_size):
            if verbose:
                print(i,end = ' ')
            lgbm = LGBMRegressor(max_depth=max_depth,num_leaves=num_leaves,learning_rate=learning_rate,**kwargs)
            self._models_list.append(
                    lgbm.fit(X = x, y = y[:,i],
                    sample_weight=sample_weight,verbose=verbose,eval_metric = eval_metric))

    def predict(self, x):
        """Empty docstring"""
        y_out = []
        for i in range(self.y_size):
            y_out.append(self._models_list[i].predict(np.array(x).reshape(1,-1))[0])
        return np.array(y_out)