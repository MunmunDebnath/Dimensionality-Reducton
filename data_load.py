#from tensorflow.keras.datasets import mnist,fashion_mnist
from scipy.io import mmread
import csv
import numpy as np
from sklearn.manifold import TSNE , Isomap
import umap
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from apply_dr_technique import applyDR

def data_loader(data, train_size, out_of_sample_size, new_out_of_sample_size, dr_technique, timeseries_projection, numOfComponent):
    # load data according to the input
    #
    # Arguments:
    #   data: dataset name
    #   train_size: size of train data
    #   out_of_sample_size: size of out-of-sample data
    #   new_out_of_sample_size: size of additional out-of-sample, to be used for update projection
    #   dr_technique: DR technique
    #   timeseries_projection: when True it is time series projection
    #   numOfComponent: number of component in low dimesional projection

    if data == "mnist":
        
        X, Y = load_mnist()
        default_train_size=2000
        default_out_of_sample_size=2000
        default_new_out_of_sample_size=1000
    elif data == "fashion_mnist":
        
        X, Y = load_fashion_mnist()
        default_train_size=2000
        default_out_of_sample_size=2000
        default_new_out_of_sample_size=1000
    elif data == "bbc":
        X, Y = load_bbc()
        default_train_size=1100
        default_out_of_sample_size=800
        default_new_out_of_sample_size=325
    elif data == "coil20":
        X, Y = load_coil20()
        default_train_size=700
        default_out_of_sample_size=600
        default_new_out_of_sample_size=140
    elif data == "spambase":
        X, Y = load_spambase()
        default_train_size=2000
        default_out_of_sample_size=1500
        default_new_out_of_sample_size=1100
    elif data == "air_quality":
        X, Y = load_air_quality(dr_technique, timeseries_projection, numOfComponent)
        default_train_size=4500
        default_out_of_sample_size=3500
        default_new_out_of_sample_size=1200
    elif data == "survival_data":
        X, Y = load_survival_data(dr_technique, timeseries_projection, numOfComponent)
        default_train_size=4000
        default_out_of_sample_size=3000
        default_new_out_of_sample_size=2000
    elif data == "iris":
        X, Y = load_iris()
        default_train_size=80
        default_out_of_sample_size=50
        default_new_out_of_sample_size=20
    else:
        print("Wrong data")
        exit
    if ((train_size+out_of_sample_size+new_out_of_sample_size)>X.shape[0]) or train_size==0 or out_of_sample_size == 0:
        train_size = default_train_size
        out_of_sample_size = default_out_of_sample_size
        new_out_of_sample_size = default_new_out_of_sample_size
    
    if timeseries_projection == True:
        train_size = 1000
        out_of_sample_size = 15
        new_out_of_sample_size = 5
   
    shuffle_idx = np.arange(0,X.shape[0])
    np.random.shuffle(shuffle_idx)
    np.random.shuffle(shuffle_idx)

    X = X[shuffle_idx]
    X_train = X[:train_size]
    X_test = X[train_size: train_size + out_of_sample_size]
    X_new = X[train_size + out_of_sample_size: train_size + out_of_sample_size + new_out_of_sample_size]

    if timeseries_projection == True:
        Y_train =  Y_test = Y_new = None
    else:
        Y = Y[shuffle_idx]
        Y_train = Y[:train_size]
        Y_test = Y[train_size: train_size + out_of_sample_size]
        Y_new = Y[train_size + out_of_sample_size: train_size + out_of_sample_size + new_out_of_sample_size]
    
    X = Y = None
    
    return X_train, Y_train, X_test, Y_test, X_new, Y_new

def load_mnist():
    # load MNIST dataset
    from tensorflow.keras.datasets import mnist
    (X, Y) , (_, _)= mnist.load_data()
    X = (X.reshape(X.shape[0],X.shape[1]*X.shape[2]))/255.0
    return X, Y

def load_fashion_mnist():
    # load Fashion MNIST dataset
    from tensorflow.keras.datasets import fashion_mnist
    (X, Y) , (_, _)= fashion_mnist.load_data()
    X = (X.reshape(X.shape[0],X.shape[1]*X.shape[2]))/255.0
    return X, Y

def load_bbc():
    # load BBC-News dataset
    X= mmread('Data/bbc/X_bbc.mtx')
    X=X.todense()
    label = []
    with open('Data/bbc/y_bbc.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            label.append(np.array(row))
    label = np.array(label)
    Y=np.zeros([label.shape[0]])
    for i in range(label.shape[0]):
        Y[i]=float(label[i,1])
    X = X.T
    return X, Y

def load_coil20():
    #load Coil20 dataset
    data = []
    with open('Data/coil20/X_coil20.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    X = np.zeros([data.shape[0], data.shape[1]])
    for i in range (data.shape[0]):
        for j in range (data.shape[1]):
            X[i,j] = np.float(data[i,j])

    label = []
    with open('Data/coil20/y_coil20.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            label.append(np.array(row))
    label = np.array(label)
    Y=np.zeros([label.shape[0]])
    for i in range(label.shape[0]):
        Y[i]=float(label[i])
    return X, Y

def load_spambase():
    # load Spambase dataset
    data = []
    with open('Data/spambase.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)

    X = np.zeros([data.shape[0], data.shape[1]])
    for i in range (data.shape[0]):
        for j in range (data.shape[1]):
            X[i,j] = np.float(data[i,j])
    Y = X[:,-1]
    X=X[:,:-1]
    return X, Y

def load_air_quality(dr_technique, timeseries_projection, numOfComponent):
    # load Air quality dataset
    data = []
    with open('Data/AirQualityUCI.csv',newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    data = data[:,2:]
    X = np.zeros([data.shape[0], data.shape[1]])
    for i in range (data.shape[0]):
        for j in range (data.shape[1]):
            X[i,j] = np.float(data[i,j].replace(",","."))
    if timeseries_projection==True:
        Y = None
    else:
        Y = assign_label(X, dr_technique, 10, numOfComponent)
    return X, Y

def load_survival_data(dr_technique, timeseries_projection, numOfComponent):
    # load Survival dataset
    data = []
    with open('Data/SurvivalData.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    data = data[:,1:]
    X = np.zeros([data.shape[0], data.shape[1]])
    for i in range (data.shape[0]):
        for j in range (data.shape[1]):
            X[i,j] = np.float(data[i,j])
    if timeseries_projection==True:
        Y = None
    else:
        Y = assign_label(X, dr_technique, 15, numOfComponent)
    return X, Y

def load_iris():
    #load IRIS dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    return X, Y

def assign_label(X, dr_technique, numberOfClasses, numOfComponent):
    # assign label to dataset, used only for better visualization with different colours for different classes
    # it does not have any influence on calculation of projection
    
    if dr_technique == "UMAP" or dr_technique == "Isomap" or dr_technique == "t-SNE":
        result = applyDR(X, dr_technique, numOfComponent)
    else:
        print("Wrong DR technique")
        exit
    result = (result- np.min(result,axis=0))/(np.max(result,axis=0)- np.min(result,axis=0))
    gmm = GaussianMixture(n_components=numberOfClasses, random_state=0).fit(result)
    Y = gmm.predict(result)
    return Y