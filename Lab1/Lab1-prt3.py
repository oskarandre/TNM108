from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

#print(cancer.DESCR)

#print(len(cancer.data[cancer.target==1]))


# # 3 columns each containing 10 figures, total 30 features
# fig,axes = plt.subplots(10,3, figsize=(12, 9))
# malignant=cancer.data[cancer.target==0] # define malignant
# benign=cancer.data[cancer.target==1] # define benign
# ax=axes.ravel() # flat axes with numpy ravel
# for i in range(30):
#  _,bins=np.histogram(cancer.data[:,i],bins=40)
#  ax[i].hist(malignant[:,i],bins=bins,color='r',alpha=.5) # red color for malignant class
#  ax[i].hist(benign[:,i],bins=bins,color='g',alpha=0.3 ) # alpha is for transparency in the
#  # overlapped region
#  ax[i].set_title(cancer.feature_names[i],fontsize=9)
#  # the x-axis coordinates are not so useful as we just want to look how well separated
#  # the histograms are
#  ax[i].axes.get_xaxis().set_visible(False)
#  ax[i].set_yticks(())
# ax[0].legend(['malignant','benign'],loc='best',fontsize=8)
# plt.tight_layout() # let's make good plots
# plt.show()


# just convert the scikit learn data-set to pandas data-frame.
cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)

# plt.subplot(1,2,1) # first plot
# plt.scatter(cancer_df['worst symmetry'], cancer_df['worst texture'],
#  s=cancer_df['worst area']*0.05, color='magenta', label='check',alpha=0.3)
# plt.xlabel('Worst Symmetry',fontsize=12)
# plt.ylabel('Worst Texture',fontsize=12)
# plt.subplot(1,2,2) # 2nd plot
# plt.scatter(cancer_df['mean radius'], cancer_df['mean concave points'],
#  s=cancer_df['mean area']*0.05, color='purple',label='check', alpha=0.3)
# plt.xlabel('Mean Radius',fontsize=12)
# plt.ylabel('Mean Concave Points',fontsize=12)
# plt.tight_layout()
# plt.show()

scaler=StandardScaler() #instantiate
# compute the mean and standard which will be used in the next command
scaler.fit(cancer.data)
X_scaled=scaler.transform(cancer.data)
# we check the minimum and maximum of the scaled features which we expect to be 0 and 1
print("after scaling minimum", X_scaled.min(axis=0))

#   PCA  
pca=PCA(n_components=2)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)
print("shape of X_pca", X_pca.shape) # let's check the shape of X_pca array

# Q1
ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)

#Q2
Xax=X_pca[:,0]
Yax=X_pca[:,1]
#Yax=X_pca[:,2]
labels=cancer.target
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,label=labl[l],marker=marker[l],alpha=alpha[l])
plt.xlabel("First Principal Component",fontsize=14)

plt.ylabel("Second Principal Component",fontsize=14)
#plt.ylabel("Third Principal Component",fontsize=14)
plt.legend()
plt.show()

#Q3
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
plt.tight_layout()
plt.show()

# correlation plot - worst features
feature_worst=list(cancer_df.columns[20:31]) # select the 'worst' features
import seaborn as sns
s=sns.heatmap(cancer_df[feature_worst].corr(),cmap='coolwarm')
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()

