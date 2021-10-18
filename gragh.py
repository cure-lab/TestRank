
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.manifold import TSNE
import math

np.random.seed(0) # make sure the graph position is reproduceble

# show  
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):   
    import matplotlib.pyplot as plt   
      
    for i in range(Mat_Label.shape[0]):  
        if int(labels[i]) == 0:    
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dr')    
        elif int(labels[i]) == 1:    
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Db')  
        else:  
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dy')  
      
    for i in range(Mat_Unlabel.shape[0]):  
        if int(unlabel_data_labels[i]) == 0:    
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'or')    
        elif int(unlabel_data_labels[i]) == 1:    
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'ob')  
        else:  
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'oy')  
      
    plt.xlabel('X1'); plt.ylabel('X2')   
    plt.xlim(0.0, 12.)  
    plt.ylim(0.0, 12.)  
    plt.show()   


def loadCircleData(num_data):  
    center = np.array([5.0, 5.0])  
    radiu_inner = 2  
    radiu_outer = 4  
    num_inner = int(num_data / 3 )
    num_outer = num_data - num_inner  
      
    data = []  
    theta = 0.0  
    for i in range(num_inner):  
        pho = (theta % 360) * math.pi / 180  
        tmp = np.zeros(2, np.float32)  
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]  
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]  
        data.append(tmp)  
        theta += 2  
      
    theta = 0.0  
    for i in range(num_outer):  
        pho = (theta % 360) * math.pi / 180  
        tmp = np.zeros(2, np.float32)  
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]  
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]  
        data.append(tmp)  
        theta += 1  
      
    Mat_Label = np.zeros((2, 2), np.float32)  
    Mat_Label[0] = center + np.array([-radiu_inner + 0.5, 0])  
    Mat_Label[1] = center + np.array([-radiu_outer + 0.5, 0])  
    labels = [0, 1]  
    Mat_Unlabel = np.vstack(data)  
    return Mat_Label, labels, Mat_Unlabel  


def propogate(latents, positive_nodes, negative_nodes):
    other_nodes = list(set(range(latents.shape[0])) - set(positive_nodes) - set(negative_nodes))
    label_prop_model = LabelPropagation(kernel='knn', n_neighbors=100, max_iter=2000)
    labels = np.zeros(latents.shape[0])
    labels[positive_nodes] = 1 
    labels[other_nodes] = -1
    # print (labels)
    label_prop_model.fit(latents, labels)
    output_labels = label_prop_model.transduction_
    output_distribution = label_prop_model.label_distributions_
    return output_labels, output_distribution



if __name__ == '__main__':
    num_unlabel_samples = 800  
    Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)  
    data = np.vstack((Mat_Label, Mat_Unlabel))
    unlabeled = np.zeros(Mat_Unlabel.shape[0])
    unlabeled.fill(-1)
    show(Mat_Label, labels, Mat_Unlabel, unlabeled) 

    positive_nodes = [1]
    negative_nodes = [0]

    other_nodes = list(set(range(data.shape[0])) - set(positive_nodes) - set(negative_nodes))
    label_prop_model = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=2000)
    labels = np.zeros(data.shape[0])
    labels[positive_nodes] = 1 
    labels[other_nodes] = -1
    print (labels)
    label_prop_model.fit(data, labels)
    output_labels = label_prop_model.transduction_


    show(Mat_Label, labels, Mat_Unlabel, output_labels[2:])  