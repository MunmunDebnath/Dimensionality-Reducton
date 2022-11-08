Source code for Reproducing, Extending and Updating Dimensionality Reductions

Reference: 
    Debnath, Munmun (2021). Reproducing, Extending and Updating Dimensionality Reductions.
    Masterarbeit. University of Stuttgart.


Execute run.py to perform reproduction of dimensionality reductions

Either modify the hyperparameteres before execution, or default values will be considered
    data: possible values are "mnist", "fashion_mnist", "bbc", "coil20", 
          "spambase", "air_quality", "survival_data", "iris"
    dr_technique: possible values are "Isomap", "t-SNE", "UMAP"
    train_size: number of data to be considered in training
                if train_size == 0 then default sample data size will be considered
    out_of_sample_size: number of unseen data to be considered 
                        for reproducing the projection of original dr_technique
    new_out_of_sample_size: number of unseen data to be considered for updating the
                            projection for out-of-sample data
    evaluate_reproduction: True or False
                           If evaluation is required, set evaluate_reproduction to True 
                           else False
    numOfNeighbour: number of nearest neighbours to be considered
    timeseries_projection: True or False
                           It can be set as True for time series data 
                           "air_quality", "survival_data".
                           If True, reproduce the projection of "MDS" 
                           for out-of-sample data and compare the result 
                           with actual projection of "MDS" 
    

Command:
    python run.py

Data:
    Datasets are present in Data directory except mnist, fashion_mnist and iris

Result:
    Projection and Evaluation results will be stored in Result directory

Dependencies:
      python = 3.9.7
      matplotlib = 3.4.3
      numpy = 1.19.5
      scikit-learn = 1.0
      scipy = 1.7.1
      tensorflow-cpu = 2.6.0 (used only for the datasets MNIST and Fashion-MNIST)
      umap-learn = 0.5.1
