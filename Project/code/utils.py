import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools as it

ANALYSIS_DIR_ROOT = "../convincingDirectory/"
words = ["Begin","Choose","Connection","Navigation","Next","Previous","Start","Stop","Hello","Web"]

def PlotConfusionMatrix(y_true,y_pred,labels,filename,title="Confusion Matrix"):
	cm = confusion_matrix(y_true,y_pred,labels)
	total = np.sum(cm,axis=1)
	total = np.array([1 if x==0 else x for x in total])
	cm_accuracy = (cm.T/total).T
	cmap = plt.get_cmap('Blues')
	plt.imshow(cm_accuracy, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	plt.xticks(labels,words,rotation=60)
	plt.yticks(labels,words)
	thresh = cm_accuracy.max()/2
	for i,j in it.product(labels,labels):
		plt.text(j,i,"{:0.2f}".format(cm_accuracy[i,j]) if cm_accuracy[i,j]>0 else "",horizontalalignment='center',color="white" if cm_accuracy[i, j]>=thresh else "black")
	plt.savefig(ANALYSIS_DIR_ROOT+filename)
	plt.show()

def PlotLoss(train_loss,test_loss,filename):
	"""
	accepts list of train and test_loss
	"""
	x = np.arange(len(train_loss))
	plt.plot(x,train_loss,label="train loss")
	plt.plot(x,test_loss,label="test loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.legend()
	plt.title("Train and Test Loss")
	plt.savefig(ANALYSIS_DIR_ROOT+filename)
	plt.show()

def PlotAccuracy(train_acc,test_acc,filename):
	"""
	accepts list of train and test accuracy
	"""
	x = np.arange(len(train_acc))
	plt.plot(x,train_acc,label="train accuracy")
	plt.plot(x,test_acc,label="test accuracy")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.legend()
	plt.title("Train and Test Accuracy")
	plt.savefig(ANALYSIS_DIR_ROOT+filename)
	plt.show()