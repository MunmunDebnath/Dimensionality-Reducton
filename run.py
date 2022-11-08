from reproduce_projection import reproduceProjection

# set data set
data = "iris" 
#  possible values are:
#  "mnist" "fashion_mnist" "bbc" "coil20" 
#  "spambase" "air_quality" "survival_data" "iris"

# set dr_technique
dr_technique = "Isomap"
# possible values are:
# "Isomap" "t-SNE" "UMAP"

# set sample size
train_size = 0 # if train_size == 0 then default sample data size will be considered
out_of_sample_size =0
new_out_of_sample_size = 0

# set evaluate_reproduction 
# if evaluation is required then set it as True, else as False
evaluate_reproduction = True # True or False

# set number of nearest neighbours to be considered
numOfNeighbour = 2 # 2 or 3

# set timeseries_projection
# if it is time curve projection then set it as True, else as False
timeseries_projection = False # True or False


# validation for time series data projection
# timeseries projection is possible for time series data "air_quality" "survival_data"
if timeseries_projection and (data == "air_quality" or data=="survival_data"):
    dr_technique = "MDS"
else:
    timeseries_projection = False

# call the main function to reproduce the projection
reproduceProjection(data, dr_technique, train_size,out_of_sample_size, new_out_of_sample_size, 
                    evaluate_reproduction, timeseries_projection, numOfNeighbour)
                    