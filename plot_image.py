import numpy as np
import matplotlib.pyplot as plt

def plotImage(X_org_low, X_project_low, Y_org, data, dr_technique, projectionType="new"):
    # plot 2D image
    #
    # Arguments:
    #  X_org_low: low dimensional projection by original DR techinque
    #  X_project_low: low dimensional projection by the proposed algorithm 
    #  Y_org: labels
    #  data: dataset name
    #  dr_technique: DR technique name
    #  projectionType: new or updated 

    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    target_names = np.unique(Y_org)
    X_org_low = (X_org_low -np.min(X_org_low,axis=0))/(np.max(X_org_low,axis=0)-np.min(X_org_low,axis=0))
    X_project_low = (X_project_low -np.min(X_project_low,axis=0))/(np.max(X_project_low,axis=0)-np.min(X_project_low,axis=0))  
    colors = "red","green","blue","yellow","magenta","black","orange","gray","cyan","brown", "olive","crimson","orangered","coral","lime","purple","darkslategray","darkcyan","plum","chocolate"
    colors = colors [:len(target_names)]
    ax1.set_title(dr_technique +" on "+projectionType+" out of sample data")
    for i, c in zip(target_names, colors):
        ax1.scatter(X_org_low[Y_org == i, 0], X_org_low[Y_org == i, 1], c=c, s=[10])
    
    ax2.set_title("Proposed algorithm on "+projectionType+" out of sample data")
    for i, c in zip(target_names, colors):
        ax2.scatter(X_project_low[Y_org == i, 0], X_project_low[Y_org == i, 1], c=c, s=[10])
    
    plt.savefig("Result\\"+data+"_"+dr_technique+"_"+projectionType+".png")

def plotTimeSeriesData(X_org_low, X_project_low, data, dr_technique, projectionType="new"):
    # plot 2D image for time series data indicating time
    #
    # Arguments:
    #  X_org_low: low dimensional projection by original DR techinque
    #  X_project_low: low dimensional projection by the proposed algorithm 
    #  data: dataset name
    #  dr_technique: DR technique name
    #  projectionType: new or updated 

    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    X_org_low = (X_org_low -np.min(X_org_low,axis=0))/(np.max(X_org_low,axis=0)-np.min(X_org_low,axis=0))
    X_project_low = (X_project_low -np.min(X_project_low,axis=0))/(np.max(X_project_low,axis=0)-np.min(X_project_low,axis=0))  

    ax1.set_title(dr_technique +" on "+projectionType+" out of sample data")
    ax1.plot(X_org_low[:, 0], X_org_low[:, 1], 'o-')
    for i in range (X_org_low.shape[0]):
        ax1.text(x=X_org_low[i,0],y=X_org_low[i,1],s=str(i))

    ax2.set_title("Proposed algorithm on "+projectionType+" out of sample data")
    ax2.plot(X_project_low[:,0], X_project_low[:,1], 'o-')
    for j in range (X_project_low.shape[0]):
        ax2.text(x=X_project_low[j,0],y=X_project_low[j,1],s=str(j))

    plt.savefig("Result\\"+data+"_time_series_"+dr_technique+"_"+projectionType+".png")

def plotShepardDiagram(D_high_scaled, D_org_low_scaled, D_project_low_scaled, data, dr_technique, projectionType):
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    fig.suptitle('Shepard Diagram', fontsize=10)
    ax1.set_title(dr_technique +" on "+projectionType+" out of sample data")
    ax1.scatter(D_high_scaled.ravel(), D_org_low_scaled.ravel(), s=0.1, c='orange')
    ax1.plot([0, 1], [0, 1], alpha=0.3, c='red')
    ax2.set_title("Proposed algorithm on "+projectionType+" out of sample data")
    ax2.scatter(D_high_scaled.ravel(), D_project_low_scaled.ravel(), s=0.1, c='orange')
    ax2.plot([0, 1], [0, 1], alpha=0.3, c='red')
    plt.savefig("Result\\"+data+"_"+dr_technique+"_"+projectionType+"_shepard_diagram.png")

    

