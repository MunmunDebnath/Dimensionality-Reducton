# References
#
# [1] Zhang, H., Wang, P., Gao, X., Qi, Y., & Gao, H. (2021). 
#     Out-of-sample data visualization using bi-kernel t-SNE. 
#     Information Visualization, 20(1), 20-34.
#     https://github.com/zhanghaily/bikernel-t-SNE/blob/main/Q_metrics.py
#
# [2] Espadoto, M., Martins, R. M., Kerren, A., Hirata, N. S., & Telea, A. C. (2019). 
#     Toward a quantitative survey of dimension reduction techniques. 
#     IEEE transactions on visualization and computer graphics, 27(3), 2153-2173.
#     https://github.com/mespadoto/proj-quant-eval/blob/master/code/01_data_collection/metrics.py
#
# Code for the normalized stress,trustworthiness, continuity and Sperman correlation  are taken from reference [1]
# Code for the Shepard diagram is taken from reference [2]

from plot_image import plotShepardDiagram
import numpy as np
import scipy.spatial.distance as dis
from scipy.stats import spearmanr

def evaluationMetric(X_high, X_org_low, X_project_low, data, dr_technique, projectionType="new", numOfNeighbour=10):
    # preform evaluation according to the input
    #
    # Arguments:
    #   X_high: high dimensional data points
    #   X_org_low: low dimensional data points are calculated by original DR technique
    #   X_project_low: low dimensional data points are calculated by proposed algorithm
    #   data: name of the dataset
    #   dr_technique: DR techinique
    #   projectionType: if first projection then "new", else "update" 
    #   numOfNeighbour: number of nearest neighbours to be considered in evaluation

    D_high = distance_matrix(X_high)
    X_org_low = (X_org_low -np.min(X_org_low,axis=0))/(np.max(X_org_low,axis=0)-np.min(X_org_low,axis=0))
    X_project_low = (X_project_low -np.min(X_project_low,axis=0))/(np.max(X_project_low,axis=0)-np.min(X_project_low,axis=0))    
    D_org_low = distance_matrix(X_org_low)
    D_project_low = distance_matrix(X_project_low)
    
    trustworthiness_org = trustworthiness(D_high, D_org_low, numOfNeighbour)
    trustworthiness_project = trustworthiness(D_high, D_project_low, numOfNeighbour)

    continuity_org = continuity(D_high, D_org_low, numOfNeighbour)
    continuity_project = continuity(D_high, D_project_low, numOfNeighbour)

    normalized_stress_org = normalized_stress(D_high, D_org_low)
    normalized_stress_project = normalized_stress(D_high, D_project_low)

    sperman_correlation_org = sperman_correlation(D_high, D_org_low)
    sperman_correlation_project = sperman_correlation(D_high, D_project_low)

    if projectionType == "new":
        mode = "w"
    else:
        mode = "a"

    file = open("Result\\"+data+"_"+dr_technique+"_evaluation.txt",mode)
    file.write("Data "+ data+"\n")
    file.write(dr_technique +": Trustworthiness "+ projectionType+ "  out of sample data "+ str(np.round(trustworthiness_org,2)) + "\n")
    file.write("Proposed algorithm: Trustworthiness "+ projectionType+ "  out of sample data "+ str(np.round(trustworthiness_project,2)) + "\n")
    file.write(dr_technique +": Continuity "+ projectionType+ "  out of sample data "+ str(np.round(continuity_org,2)) + "\n")
    file.write("Proposed algorithm: Continuity "+ projectionType+ "  out of sample data "+ str(np.round(continuity_project,2)) + "\n")
    file.write(dr_technique +": Normalized Stress "+ projectionType+ "  out of sample data "+ str(np.round(normalized_stress_org,2)) + "\n")
    file.write("Proposed algorithm: Normalized Stress "+ projectionType+ "  out of sample data "+ str(np.round(normalized_stress_project,2)) + "\n")
    file.write(dr_technique +": Sperman Correlation "+ projectionType+ "  out of sample data "+ str(np.round(sperman_correlation_org,2)) + "\n")    
    file.write("Proposed algorithm: Sperman Correlation "+ projectionType+ "  out of sample data "+ str(np.round(sperman_correlation_project,2)) + "\n")
    file.write("\n\n\n")
    file.close()

    shepard_diagram(D_high, D_org_low, D_project_low, data, dr_technique, projectionType)

def distance_matrix(X):
    # calculate distance matrix
    #
    # Arguments:
    #   X: dataset

    distance = dis.squareform(dis.pdist(X))
    return distance

def normalized_stress(D_high, D_low):
    # calculate normalized stress
    #
    # Arguments:
    #   D_high: distance matrix of high dimensional data points
    #   D_low: distance matrix of low dimensional data points

    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)

def trustworthiness(D_high,D_low,k):
    # calculate trustworthiness
    #
    # Arguments:
    #   D_high: distance matrix of high dimensional data points
    #   D_low: distance matrix of low dimensional data points
    #   k: number of nearest neighbours
   
    N = D_high.shape[0]
    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()
    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]
    sum_i = 0
    for i in range(N):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])
        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k
        sum_i += sum_j
    return float((1 - (2 / (N * k * (2 * N - 3 * k - 1)) * sum_i)).squeeze())

def continuity(D_high,D_low,k):
    # calculate continuty
    #
    # Arguments:
    #   D_high: distance matrix of high dimensional data points
    #   D_low: distance matrix of low dimensional data points
    #   k: number of nearest neighbours

    N = D_high.shape[0]
    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()
    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]
    sum_i = 0
    for i in range(N):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])
        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k
        sum_i += sum_j
    return float((1 - (2 / (N * k * (2 * N - 3 * k - 1)) * sum_i)).squeeze())

def sperman_correlation(D_high, D_low):
    # calculate sperman correlation
    #
    # Arguments:
    #   D_high: distance matrix of high dimensional data points
    #   D_low: distance matrix of low dimensional data points

    correlation = spearmanr(D_high, D_low,axis=None)[0]
    return correlation

def shepard_diagram(D_high, D_org_low, D_project_low, data, dr_technique, projectionType):
    # plot shepard diagram
    #
    # Arguments:
    #   D_high: distance matrix of high dimensional data points
    #   D_org_low: distance matrix of low dimensional data points,
    #              where low dimensional data points are calculated by original DR technique
    #   D_project_low: distance matrix of low dimensional data points,
    #                  where low dimensional data points are calculated by proposed algorithm
    #   data: name of the dataset
    #   dr_technique: DR techinique
    #   projectionType: if first projection then "new", else "update" 

    D_high_scaled=D_high/np.max(D_high)
    D_org_low_scaled=D_org_low/np.max(D_org_low)
    D_project_low_scaled=D_project_low/np.max(D_project_low)
    plotShepardDiagram(D_high_scaled, D_org_low_scaled, D_project_low_scaled, data, dr_technique, projectionType)