from apply_dr_technique import applyDR

def train(X, dr_technique, numOfComponent=2):
    # return low dimensional projection of the input data points according to given input
    #
    # Arguments:
    #   X: dataset
    #   dr_technique: DR technique
    #   numOfComponent: number of dimensions for low dimension projection 
    
    X_train_low = applyDR(X, dr_technique, numOfComponent)
    return X_train_low