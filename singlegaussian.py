import os
import cv2
import numpy as np
import math
from numpy.linalg import inv, det

from scipy.stats import norm
from sklearn.metrics import auc, roc_curve, confusion_matrix
from scipy.stats import multivariate_normal


import matplotlib.pyplot as plt
import numpy.matlib

from utils import get_files_list, dataPreparation


images_dir = "originalPics/"
fddb_file_path = "FDDB-folds/"
img_format = ".jpg"


dirPath = ""
outputDir = "output/"
bboxfddfolds = "bbox-FDDB-folds/"
bboxfacesdir = "faces/"
bboxnonfacesdir = "non_faces/"
facedim = (20, 20)

modelpath = "models/"


facedatatr, nonfacedatatr, facelabelstr, tr_non_face_labels, facedatate, nonfacedatate, facelabelste, nonfacelabelste = dataPreparation()


class SingleGaussian():
    def __init__(self, facedatatr, nonfacedatatr, facelabelstr, tr_non_face_labels, \
                 facedatate, nonfacedatate, facelabelste, nonfacelabelste, vec_len, img_dim=(20,20,3)):
        self.facedatatr = facedatatr
        self.nonfacedatatr = nonfacedatatr
        self.facelabelstr = facelabelstr
        self.nonfacelabelstr = tr_non_face_labels
        
        self.facedatate = facedatate
        self.nonfacedatate = nonfacedatate
        self.testing_data = np.concatenate((self.facedatate, self.nonfacedatate), axis=0)
        
        self.facelabelste = facelabelste
        self.nonfacelabelste = nonfacelabelste
        
        self.vec_len = vec_len
        self.img_dim = img_dim
        
        self.storing_dir = "Single_Gaussian/"
        
    def fit(self):
        self.tr_f_mu = self.facedatatr.mean(axis=0)
        self.tr_nf_mu = self.nonfacedatatr.mean(axis=0)
        
        self.tr_face_sigma = np.cov(self.facedatatr, rowvar=False, bias=1, ddof=None)
        self.tr_face_sigma = np.diagonal(self.tr_face_sigma)
        self.tr_f_covariance = np.diag(self.tr_face_sigma, 0)
        
        self.tr_non_face_sigma = np.cov(self.nonfacedatatr, rowvar=False, bias=1, ddof=None)
        self.tr_non_face_sigma = np.diagonal(self.tr_non_face_sigma)
        self.tr_nf_covariance = np.diag(self.tr_non_face_sigma, 0)
                
    def pdf(self, data, mean, covs):
        print(data.shape, mean.shape, covs.shape)
        temp1 = np.matmul((data[:,0].reshape(-1,1)-mean[0]).T, inv(covs[0]))
        temp2 = -0.5*np.matmul(temp1, data[:,0].reshape(-1,1)-mean[0])
        pdf = np.exp(temp2)/(np.sqrt(det(covs[k]) * (2*np.pi**data.shape[0])))
        return pdf
        
    def predict(self):
        self.pred_labels = []
        self.pred_scores = []
        
        self.face_pdf = multivariate_normal.pdf(self.facedatate, self.tr_f_mu, self.tr_f_covariance)
        self.non_face_pdf = multivariate_normal.pdf(self.nonfacedatate, self.tr_nf_mu, self.tr_nf_covariance) 
        
        self.face_predict_labels = [1 if i>0.5 else 0 for i in self.face_pdf]
        self.non_face_predict_labels = [0 if i>0.5 else 1 for i in self.non_face_pdf]
        
        self.pred_labels = self.face_predict_labels + self.non_face_predict_labels
        self.pred_scores = np.concatenate((self.face_pdf, self.non_face_pdf), axis=0)
    
    def visualize_mean_and_cov(self):
        self.mean_f_image = self.tr_f_mu.reshape(self.img_dim)
        self.mean_f_image =  self.mean_f_image*(255/np.max(self.mean_f_image))

        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Mean_Face_Image.jpg", self.mean_f_image)


        cov_f_image = np.diag(self.tr_f_covariance)
        cov_f_image = cov_f_image/np.max(cov_f_image)
        self.cov_f_image = cov_f_image.reshape(self.img_dim)
        self.cov_f_image =  self.cov_f_image*(255/np.max(self.cov_f_image))
        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Cov_Face_Image.jpg", self.cov_f_image)
        
        self.mean_nf_image = self.tr_nf_mu.reshape(self.img_dim)
        self.mean_nf_image =  self.mean_nf_image*(255/np.max(self.mean_nf_image))

        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Mean_Non_Face_Image.jpg", self.mean_nf_image)

        cov_nf_image = np.diag(self.tr_nf_covariance)
        cov_nf_image = cov_nf_image/np.max(cov_nf_image)
        self.cov_nf_image = cov_nf_image.reshape(self.img_dim)
        self.cov_nf_image =  self.cov_nf_image*(255/np.max(self.cov_nf_image))
        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Cov_Non_Face_Image.jpg", self.cov_nf_image)
                
            
        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Mean_Face_Image_resize.jpg", cv2.resize(self.mean_f_image, (60,60)))
        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Mean_Non_Face_Image_resize.jpg", cv2.resize(self.mean_nf_image, (60,60)))
        
        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Cov_Face_Image_resize.jpg", cv2.resize(self.cov_f_image, (60,60)))
        cv2.imwrite(dirPath + modelpath + self.storing_dir + "Cov_Non_Face_Image_resize.jpg", cv2.resize(self.cov_nf_image, (60,60)))
            
    def create_confusion_matrix(self):
        self.testing_data_labels = np.concatenate((single_gaussian.facelabelste, single_gaussian.nonfacelabelste), axis=0)
    
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(self.pred_labels)): 
            if self.testing_data_labels[i]==self.pred_labels[i]==1:
               TP += 1
            if self.pred_labels[i]==1 and self.testing_data_labels[i]!=self.pred_labels[i]:
               FP += 1
            if self.testing_data_labels[i]==self.pred_labels[i]==0:
               TN += 1
            if self.pred_labels[i]==0 and self.testing_data_labels[i]!=self.pred_labels[i]:
               FN += 1

        return(TP, FP, TN, FN)
        
    def plot_roc_curve(self):
        fpr, tpr, threshold = roc_curve(self.testing_data_labels, self.pred_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(dirPath + modelpath + self.storing_dir + "Single_Gaussian_ROC_Curve.png")


single_gaussian = SingleGaussian(facedatatr, nonfacedatatr, facelabelstr, tr_non_face_labels, \
                                 facedatate, nonfacedatate, facelabelste, nonfacelabelste,\
                                1200, (20,20,3))


single_gaussian.fit()

single_gaussian.predict()


TP, FP, TN, FN = single_gaussian.create_confusion_matrix()

fpr =  FP/(FP+TN)
fnr = FN/(TP+FN)
mis_class_rate = (FP + FN)/200
print("FPR rate: {}".format(fpr))
print("FNR rate: {}".format(fnr))
print("Misclassification rate: {}".format(mis_class_rate))


single_gaussian.plot_roc_curve()


single_gaussian.visualize_mean_and_cov()