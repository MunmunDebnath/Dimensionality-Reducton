from data_load import data_loader
from train import train
from out_of_sample_projection import outOfSampleProjection
from apply_dr_technique import applyDR
from plot_image import plotImage, plotTimeSeriesData
from evaluation_metric import evaluationMetric
import numpy as np

def reproduceProjection(data = "mnist", dr_technique = "UMAP", 
                        train_size = 0,out_of_sample_size = 0, new_out_of_sample_size = 0,
                        evaluate_reproduction = True, timeseries_projection = False, 
                        numOfNeighbour=2
                        ):
    # main function to reproduce the projection of a DR technique for out of sample
    #
    # Arguments:
    #   data: dataset
    #   dr_technique: DR technique
    #   train_size: size of data to be used in training
    #   out_of_sample_size: size of data to be used in reproduction
    #   new_out_of_sample_size: addtional data instances to be considered for updated projection
    #   evaluate_reproduction: when True evaluation is done based on metrics
    #   timeseries_projection: can be True only for time series data
    #   numOfNeighbour: number of nearest neighbour from train dataset to be considered

    # In this thesis data points are projected in 2 dimensions
    numOfComponent = 2

    # data load
    X_train, Y_train, X_test, Y_test, X_new, Y_new = data_loader(data, train_size, out_of_sample_size, new_out_of_sample_size, dr_technique, timeseries_projection, numOfComponent)
   
    # training
    X_train_low = train(X_train, dr_technique, numOfComponent)

    # out of sample projection
    X_unseen_low, base_matrix = outOfSampleProjection(X_test, X_train, X_train_low, numOfComponent, numOfNeighbour, updateProjection=False)

    # result of original dimensionality reduction technique, used for comparison purpose only
    X_org_unseen_low = applyDR(X_test, dr_technique, numOfComponent)

    if timeseries_projection == False :
        # plot image
        plotImage(X_org_unseen_low, X_unseen_low, Y_test, data, dr_technique, projectionType="new")        
        
        if evaluate_reproduction:
            if X_org_unseen_low.shape[0]>300:
                n_neigh = 10
            else:
                n_neigh = 5
            # evaluation metrics
            evaluationMetric(X_test, X_org_unseen_low, X_unseen_low, data, dr_technique, projectionType="new", numOfNeighbour=n_neigh)
    else:
        # plot the result
        plotTimeSeriesData(X_org_unseen_low, X_unseen_low, data, dr_technique, projectionType="new")

    # out of sample projection updation
    X_update_unseen_low, base_matrix = outOfSampleProjection(X_new, X_train, X_train_low, numOfComponent, numOfNeighbour, base_matrix, updateProjection=True)

    X_update = np.zeros([X_test.shape[0]+X_new.shape[0],X_test.shape[1]])
    X_update[:X_test.shape[0]] = X_test
    X_update[X_test.shape[0]:] = X_new

    # projection for extended out of sample by original DR, used for comparison purpose only
    X_org_update_unseen_low = applyDR(X_update, dr_technique, numOfComponent)
    
    if timeseries_projection == False :
        Y_update = np.zeros([Y_test.shape[0]+Y_new.shape[0],])
        Y_update[:Y_test.shape[0]] = Y_test
        Y_update[X_test.shape[0]:] = Y_new   
      
        # plot image
        plotImage(X_org_update_unseen_low, X_update_unseen_low, Y_update, data, dr_technique, projectionType="updated")
        
        if evaluate_reproduction:
            if X_org_update_unseen_low.shape[0]>300:
                n_neigh = 10
            else:
                n_neigh = 5
            # evaluation metrics
            evaluationMetric(X_update, X_org_update_unseen_low, X_update_unseen_low, data, dr_technique, projectionType="updated", numOfNeighbour=n_neigh)
    else:
        # plot the result
        plotTimeSeriesData(X_org_update_unseen_low, X_update_unseen_low, data, dr_technique, projectionType="updated")
