


import os
import cv2

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.special import gamma,digamma,gammaln
from scipy.optimize import fminbound

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import math
import matplotlib.pyplot as plt

import numpy.matlib
from numpy.linalg import inv, det
import numpy as np

from utils import get_files_list, dataPreparation

dir_path = ""

imagesDir = "originalPics/"
fddbfilepath = "FDDB-folds/"
imgFormat = ".jpg"

outputDir = "output/"
bboxfddfolds = "bbox-FDDB-folds/"
bboxfacesdir = "faces/"
bboxnonfacesdir = "non_faces/"
facedim = (20, 20)

model_path = "models/"


no_of_components = 5
dimensions_after_pca = 100


tr_face_data, tr_non_face_data, tr_face_labels, tr_non_face_labels, te_face_data, te_non_face_data, te_face_labels, te_non_face_labels = dataPreparation()



def preprocessing(data):
    pca = PCA(n_components=100)
    pca.fit(data)
    pca_data = pca.transform(data)
    std_scaler = StandardScaler()
    std_scaler.fit(pca_data)
    std_data = std_scaler.transform(pca_data)
    return pca, std_data



pca_f, tr_f_data = preprocessing(tr_face_data)
pca_nf, tr_nf_data = preprocessing(tr_non_face_data)
_, te_f_data = preprocessing(te_face_data)
_, te_nf_data = preprocessing(te_non_face_data)



tr_f_data, tr_nf_data, te_f_data, te_nf_data = tr_f_data.T, tr_nf_data.T, te_f_data.T, te_nf_data.T


tr_f_data.shape, tr_nf_data.shape, te_f_data.shape, te_nf_data.shape



def t_cost(v, e_of_h, e_of_log_of_h):
    length = len(e_of_h)
    t1 = (v/2) * np.log((v/2))
    t2 = gammaln((v/2))

    t_cost = 0
    for i in range(length):
       t3 = ((v/2) - 1) * e_of_log_of_h[i]
       t4 = (v/2) * e_of_h[i]
       t_cost += t1 - t2 + t3 - t4
    t_cost = -t_cost
    return t_cost



class T_Distribution():
    def __init__(self, data_size, mean, covs, v, image_dim=(20,20)):
        self.data_size = data_size
        self.mean = mean
        self.covs = covs
        self.v = v
        self.e_of_h = np.zeros(self.data_size)
        self.e_of_log_of_h = np.zeros(self.data_size)
        self.term = np.zeros(self.data_size)
        
    def update_v(self):
        v = fminbound(t_cost, 0, 10, args=(self.e_of_h, self.e_of_log_of_h)) 
        return v
        
    def EM(self, data):
        dimensions = data.shape[0]
        for row in range(self.data_size):
            temp = np.matmul((data[:,row].reshape(-1,1)-self.mean).T, inv(self.covs))
            term = np.matmul(temp, (data[:,row].reshape(-1,1)-self.mean))
            self.term[row] = term
            self.e_of_h = (self.v+dimensions) /(self.v+term)
            self.e_of_log_of_h = digamma((self.v+dimensions)/2) - np.log((self.v+term)/2)
                        
        self.mean = (np.sum(self.e_of_h * data, axis=1)/np.sum(self.e_of_h)).reshape(dimensions, 1)
        
        numerator = np.zeros((dimensions, dimensions))
        for i in range(self.data_size):
            temp = np.matmul((data[:,i].reshape(-1,1)-self.mean), (data[:,i].reshape(-1,1)-self.mean).T)
            numerator += self.e_of_h*temp
        self.covs = numerator/np.sum(self.e_of_h)
        self.covs = np.diag(np.diag(self.covs))
        
        self.v = self.update_v()
        
        for i in range(self.data_size):
            temp = np.matmul((data[:,row].reshape(-1,1)-self.mean).T, inv(self.covs))
            temp = np.matmul(temp, (data[:,row].reshape(-1,1)-self.mean))
            self.term[i] = term
            
    def predict(self, data, row):
        dimensions = self.mean.shape[0]
        term1 = gamma((self.v + dimensions)/2) / (((self.v * np.pi)** dimensions/2) *np.sqrt(det(self.covs))*gamma(self.v/2))
        term2 = np.matmul( (data[:,row].reshape(-1,1)-self.mean).T,inv(self.covs) )                                  
        term2 = np.matmul(term2,(data[:,row].reshape(-1,1) - self.mean))
        term = (1 + term2/self.v)
        prob = term1 * pow(term, -(self.v+dimensions)/2)
        return prob[0,0]
        
mean_f_data = np.mean(tr_f_data, axis=1)
mean_nf_data = np.mean(tr_nf_data, axis=1)

cov_f_data = np.cov(tr_f_data)
cov_f_data = np.diag(np.diagonal(cov_f_data), 0)
cov_nf_data = np.cov(tr_nf_data)
cov_nf_data = np.diag(np.diagonal(cov_nf_data), 0)

mean_f_data.shape, cov_f_data.shape

t_dist_f = T_Distribution(1000, mean_f_data.reshape(-1,1), cov_f_data, v=5)
t_dist_nf = T_Distribution(1000, mean_nf_data.reshape(-1,1), cov_nf_data, v=5)


for i in range(1000):
    print("\nPerforming Iteration - {}".format(i))
    print("t_dist_for_face")
    t_dist_f.EM(tr_f_data)
    print("\ntdist_for_nonface")
    t_dist_nf.EM(tr_nf_data)

print("Visualizing Mean")
mean_f_img = np.dot(t_dist_f.mean[:,0], pca_f.components_) + pca_f.mean_
mean_f_img = np.array(mean_f_img).astype('uint8')
mean_f_img = np.reshape(mean_f_img,(20,20))
plt.imshow(mean_f_img, cmap="gray")
plt.show()
print("Visualizing Covariance")
cov_f_img = np.diagonal(t_dist_f.covs)
cov_f_img = np.matmul(np.log(cov_f_img), pca_f.components_) + pca_f.mean_
cov_f_img =  cov_f_img.reshape(20,20)
plt.imshow(cov_f_img, cmap='gray')
plt.show()


cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Mean_Face_Image.jpg", mean_f_img)
cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Cov_Face_Image.jpg", cov_f_img) 

cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Mean_Face_Image_Resize.jpg",             cv2.resize(mean_f_img, (60,60)))
cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Cov_Face_Image_Resize.jpg",             cv2.resize(cov_f_img, (60,60)))

print("Visualizing Mean")
mean_nf_img = np.dot(t_dist_nf.mean[:,0], pca_nf.components_) + pca_nf.mean_
mean_nf_img = np.array(mean_nf_img).astype('uint8')
mean_nf_img = np.reshape(mean_nf_img,(20,20))
plt.imshow(mean_nf_img, cmap="gray")
plt.show()


print("Visualizing Covariance")
cov_nf_img = np.diagonal(t_dist_nf.covs)
cov_nf_img = np.dot(np.log(cov_nf_img), pca_nf.components_) + pca_nf.mean_
cov_nf_img = cov_nf_img.reshape(20,20)
plt.imshow(cov_nf_img, cmap='gray')
plt.show()

cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Mean_Non_Face_Image.jpg", mean_nf_img)
cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Cov_Non_Face_Image.jpg", cov_nf_img) 

cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Mean_Non_Face_Image_Resize.jpg",             cv2.resize(mean_nf_img, (60,60)))
cv2.imwrite(dir_path + model_path + "T_Distribution/" + "Cov_Non_Face_Image_Resize.jpg",             cv2.resize(cov_nf_img, (60,60)))

pred_f_fdata, pred_f_nfdata, pred_nf_fdata, pred_nf_nfdata = [], [], [], []

for i in range(100):
    pred_f_fdata.append(t_dist_f.predict(te_f_data, i))
    pred_f_nfdata.append(t_dist_f.predict(te_nf_data, i))
    
    pred_nf_fdata.append(t_dist_nf.predict(te_f_data, i))
    pred_nf_nfdata.append(t_dist_nf.predict(te_nf_data, i))

pred_f_fdata = np.array(pred_f_fdata)
pred_f_nfdata = np.array(pred_f_nfdata)
pred_nf_fdata = np.array(pred_nf_fdata)
pred_nf_nfdata = np.array(pred_nf_nfdata)

pred_f_fdata = pred_f_fdata/(pred_f_fdata+pred_f_nfdata)
pred_f_nfdata = pred_f_nfdata/(pred_f_fdata+pred_f_nfdata)
pred_nf_fdata = pred_nf_fdata/(pred_nf_fdata+pred_nf_nfdata)
pred_nf_nfdata = pred_nf_nfdata/(pred_nf_fdata+pred_nf_nfdata)

pred_f_labels = [1 if pred_f_fdata[i]>pred_nf_fdata[i] else 0 for i in range(100)]
pred_nf_labels = [1 if pred_f_nfdata[i]>pred_nf_nfdata[i] else 0 for i in range(100)]

pred_labels = pred_f_labels + pred_nf_labels
true_labels = [1]*100 + [0]*100

def confusion_matrix(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

TP, FP, TN, FN = confusion_matrix(true_labels, pred_labels)

fpr =  FP/(FP+TN)
fnr = FN/(TP+FN)
mis_class_rate = (FP + FN)/200
print("False Positive Rate: {}".format(fpr))
print("False Negative Rate: {}".format(fnr))
print("Mis Classification Rate: {}".format(mis_class_rate))


fpr, tpr, threshold = roc_curve([1]*100 + [0]*100, np.append(pred_f_fdata, pred_nf_fdata))
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, "g--", label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(dir_path + model_path + "T_Distribution/" + "T_Distribution_ROC_Curve.png")






