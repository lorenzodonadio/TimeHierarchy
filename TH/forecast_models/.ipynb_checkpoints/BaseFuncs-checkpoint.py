import pandas as pd
import numpy as np

from ..utils.utils import compute_y_hat

class BaseFuncs():
    def __init__(self):
        self.compute_y_hat = compute_y_hat
    
    def _compute_errors(self,to_remove = 10):
        '''Empty docstring'''
        self.error_df = pd.DataFrame()
        
        if self.use_short:
            for k in self.model_tree:
                
                for k2 in self.model_tree[k]:
                    # calculate insample errors
                    tmp =  self.hierarchy_short[k][k2][self.col_name].values - self.model_tree[k][k2].fittedvalues
                    self.error_df['e_'+str(int(k))+'_'+str(int(k2))] = tmp
        else:
            for k in self.hierarchy:
                
                tmp = self.hierarchy[k].copy()
                tmp['error'] = tmp[self.col_name] - self.model_tree[k].fittedvalues
                for i in np.unique(tmp['group_id']):
                    tmp2 = tmp[tmp['group_id']==i].reset_index(drop = True)
                    self.error_df['e_'+str(int(k))+'_'+str(int(i))] = tmp2['error']
        
            del(tmp,tmp2) #freeup some memory becasue we copied the df
            
        self.error_df = self.error_df[to_remove:]  ## remove the first in-sample errors, 
        ### unclear how to models give fitted values for the begining of the trainiing set....