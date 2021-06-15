import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error

def compute_metric(ytrue,ypred,n_nodes ,method = 'RMSE',scale_method = None):
    '''ytrue and ypred are matrices, same shape.
       n_nodes indicates the tree structure and must be coherent with the structure of ytrue,ypred'''
    n_nodes = np.sort([int(i/min(n_nodes)) for i in n_nodes]) 
    assert ytrue.shape[0] == ypred.shape[0] == sum(n_nodes)
    idx_list = [0]+list(np.cumsum(n_nodes))
    metrics = []
    for k,(i,j) in enumerate(zip(idx_list[:-1],idx_list[1:])):
        yt,yp = ytrue[i:j,:].reshape(-1,1),ypred[i:j,:].reshape(-1,1)

        if scale_method == 'node':
            scale = n_nodes[::-1][k]
            
            if method == 'RMSE':
                metrics.append(np.sqrt(mean_squared_error(yt,yp))/scale)
            elif method == 'MAE':
                metrics.append(mean_absolute_error(yt,yp)/scale)
            elif method == 'MedAE':
                metrics.append(median_absolute_error(yt,yp)/scale)
            else:
                raise ValueError('Invalid Method')
                
        elif scale_method == 'perc':
            scale = np.mean(yt)
            
            if method == 'RMSE':
                metrics.append(100*np.sqrt(mean_squared_error(yt,yp))/scale)
            elif method == 'MAE':
                metrics.append(100*mean_absolute_error(yt,yp)/scale)
            elif method == 'MedAE':
                metrics.append(100*median_absolute_error(yt,yp)/scale)
            else:
                raise ValueError('Invalid Method')
                
        else:
            if method == 'RMSE':
                metrics.append(np.sqrt(mean_squared_error(yt,yp)))
            elif method == 'MAE':
                metrics.append(mean_absolute_error(yt,yp))
            elif method == 'MedAE':
                metrics.append(median_absolute_error(yt,yp))
            else:
                raise ValueError('Invalid Method')

    return metrics