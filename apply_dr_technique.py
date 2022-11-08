from sklearn import neighbors
from sklearn.manifold import TSNE, Isomap, MDS
import umap
from sklearn.metrics.pairwise import pairwise_distances

def applyDR(X, dr_technique, numOfComponent=2):
    # execute DR technique on data according to given input
    #
    # Arguments:
    #   X: dataset
    #   dr_technique: DR technique
    #   numOfComponent: number of dimensions for low dimension projection 
    
    if (X.shape[0]<300):
        numOfNeighbors = 9
    else:
        if dr_technique == "UMAP":
            numOfNeighbors = 15
        else:
            numOfNeighbors = 30

    if dr_technique == "UMAP":
        result = umap.UMAP(n_components=numOfComponent, n_neighbors=numOfNeighbors).fit_transform(X)
    elif dr_technique == "Isomap":
        result = Isomap(n_components=numOfComponent).fit_transform(X)
    elif dr_technique == "t-SNE":
        result = TSNE(n_components=numOfComponent, perplexity=numOfNeighbors).fit_transform(X)
    elif dr_technique == "MDS":
        result = MDS(n_components=numOfComponent).fit_transform(pairwise_distances(X, metric="euclidean"))
    else:
        print("Wrong DR technique")
        exit
    return result