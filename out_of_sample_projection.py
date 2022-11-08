from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import KernelPCA
import numpy as np

def outOfSampleProjection(X_test, X_train, X_train_low, numOfComponent, numOfNeighbour, base=None, updateProjection=False):
    # calculate out-of-sample projection
    #
    # Arguments:
    #   X_test: high dimensional out-of-sample data points
    #   X_train: high dimensional training data points 
    #   X_train_low: low dimensional projection of X_train by original DR techinique
    #   numOfComponent: number of dimensions in low dimensional projection
    #   numOfNeighbour: number of nearest neighbour from train dataset to be considered
    #   base: it helps in out-of-sample projection updation
    #         during first projection it is empty and later it contains information 
    #         of earlier projections
    #   updateProjection: False for first projection, True for later projection
    #
    # Returns:
    #   X_test_low: low dimensional projection of X_test by proposed algorithm
    #   base: information of current and pervious projections

    kpca_test =KernelPCA(n_components=numOfComponent,kernel='cosine').fit_transform(X_test)
    kpca_test = (kpca_test -np.min(kpca_test,axis=0))/(np.max(kpca_test,axis=0)-np.min(kpca_test,axis=0))
    new_stack = np.zeros([kpca_test.shape[0],kpca_test.shape[1]*(numOfNeighbour+1)])
    dist_test_train =pairwise_distances(X_test, X_train, metric='sqeuclidean')
    index = np.argsort(dist_test_train,axis = 1)[:,:numOfNeighbour]
    for i in range(kpca_test.shape[0]):
        for j in range(numOfNeighbour):
            new_stack[i,j*numOfComponent:(j+1)*numOfComponent]= X_train_low[index[i,j]]
    new_stack[:,numOfNeighbour*numOfComponent:]=kpca_test

    if updateProjection==True:
        new_stack_update = np.zeros([base.shape[0]+new_stack.shape[0],base.shape[1]])
        new_stack_update[:base.shape[0]] = base
        new_stack_update[base.shape[0]:] = new_stack
        base = new_stack_update
    else:
        base = new_stack

    kernel_matrix=np.matmul(base,base.T)
    X_test_low= KernelPCA(n_components=numOfComponent,kernel='precomputed').fit_transform(kernel_matrix)

    return X_test_low, base