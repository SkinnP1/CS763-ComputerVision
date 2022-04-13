# used below link for vgg16 as feature extractor
# https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
# https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
import numpy as np
import pickle as pkl
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import PlotConfusionMatrix, PlotLoss, PlotAccuracy

DATA_DIR_ROOT = "../data/MIRACL_VC1/"
RESULTS_DIR_ROOT = "../results/"
TEST_DATA = "test_resnet50_feats_SEEN.pkl"

BATCH_SIZE = 16
TIME_STEPS = 5
N_CLASS = 10

class MIRACL_VC1_FEAT(Dataset):
	def __init__(self,data):
		self.X = data['feats']
		self.Y = data['labels']

	def __len__(self):
		return len(self.X)

	def __getitem__(self,idx):
		return [self.X[idx], self.Y[idx]]

def get_data(path):
	with open(path,'rb') as f:
		data = pkl.load(f)
	print("DATA Instances : ",len(data['labels']))
	dataset = MIRACL_VC1_FEAT(data)
	return dataset

class TOP_LSTM(nn.Module):
	def __init__(self):
		super().__init__()
		self.feat_dim = 2048

		self.lstm1 = nn.LSTM(input_size = self.feat_dim, hidden_size = 1, bidirectional = True)
		self.processOut = nn.Linear(TIME_STEPS*2,N_CLASS)
		self.classifier = nn.Softmax()

	def forward(self,input):
		out,_ = self.lstm1(input.reshape(TIME_STEPS,BATCH_SIZE,self.feat_dim))
		out = out.permute(1,0,2)
		out = torch.flatten(out,1)
		scores = self.processOut(out)
		labels = self.classifier(scores).argmax(axis=1)
		return scores,labels

print("load model")
model = TOP_LSTM()
model.load_state_dict(torch.load(RESULTS_DIR_ROOT+"resnet50_checkpoint_lstm_SEEN.p"))

test_dataset = get_data(DATA_DIR_ROOT + TEST_DATA)
test_data_dl = DataLoader(test_dataset, batch_size = BATCH_SIZE * TIME_STEPS, drop_last=True)

test_accuracy_trace = []
criterion = nn.CrossEntropyLoss()
print("Start testing")
model.eval()
test_loss = 0
test_words_pred = []
test_words_true = []
for i,(feats,labels) in enumerate(test_data_dl):
	scores,pred_labels = model(feats)
	true_labels = labels.reshape(TIME_STEPS,BATCH_SIZE)[-1,:]
	loss = criterion(scores,true_labels)
	test_loss+=loss.item()

	test_words_true = test_words_true + true_labels.tolist()
	test_words_pred = test_words_pred + pred_labels.tolist()

test_acc = accuracy_score(test_words_true,test_words_pred)
print("prediction_loss = ",test_loss/i)
print("prediction accuracy_score = ", test_acc)
print("\n")

PlotConfusionMatrix(test_words_true,test_words_pred,np.arange(N_CLASS),"pred_conf_mat.jpg",title = "pred confusion matrix")
