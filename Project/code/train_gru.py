# used below link for vgg16 as feature extractor
# https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
# https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import PlotConfusionMatrix, PlotLoss, PlotAccuracy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("feat",help="choose feature type",choices=['vgg16','resnet18','resnet50','resnet152'])
parser.add_argument("type",help="choose experiment type",choices=["SEEN","UNSEEN"],default="SEEN")
args = parser.parse_args()

DATA_DIR_ROOT = "../data/MIRACL_VC1/"
RESULTS_DIR_ROOT = "../results/"

if args.type == "SEEN":
	TRAIN_DATA = "train_" + args.feat + "_feats_SEEN.pkl"
	TEST_DATA = "test_" + args.feat + "_feats_SEEN.pkl"
	FEAT_LABEL = "feats"
if args.type == "UNSEEN":
	TRAIN_DATA = "train_" + args.feat + "_feats_UNSEEN.pkl"
	TEST_DATA = "test_" + args.feat + "_feats_UNSEEN.pkl"
	FEAT_LABEL = "resnet_feats"
	if args.feat == "vgg16":	
		TRAIN_DATA = "train_anno_with_vgg16_feats.pkl"
		TEST_DATA = "test_anno_with_vgg16_feats.pkl"
	elif args.feat == "resnet18":
		TRAIN_DATA = "train_anno_with_resnet50_feats.pkl"
		TEST_DATA = "test_anno_with_resnet50_feats.pkl"
	elif args.feat == "resnet50":
		TRAIN_DATA = "train_anno_with_resnet50_feats.pkl"
		TEST_DATA = "test_anno_with_resnet50_feats.pkl"
	elif args.feat == "resnet152":
		TRAIN_DATA = "train_anno_with_resnet50_feats.pkl"
		TEST_DATA = "test_anno_with_resnet50_feats.pkl"


if args.feat == "vgg16":
	FEAT_DIM = 12800
elif args.feat == "resnet18":
	FEAT_DIM = 512
elif args.feat == "resnet50":
	FEAT_DIM = 2048
elif args.feat == "resnet152":
	FEAT_DIM = 2048
else:
	print("choose valid feature type")

BATCH_SIZE = 16
TIME_STEPS = 5
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.00001
MOMENTUM = 0.9
EPOCHS = 300
N_CLASS = 10

class MIRACL_VC1_FEAT(Dataset):
	def __init__(self,data):
		self.X = data[FEAT_LABEL]
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

class TOP_GRU(nn.Module):
	def __init__(self):
		super().__init__()
		self.feat_dim = FEAT_DIM

		self.gru = nn.GRU(input_size = self.feat_dim , hidden_size = 1 ,bidirectional = True)
		self.processOut = nn.Linear(TIME_STEPS*2,N_CLASS)
		self.classifier = nn.Softmax()

	def forward(self,input):
		out,_ = self.gru(input.reshape(TIME_STEPS,BATCH_SIZE,self.feat_dim))
		out = out.permute(1,0,2)
		out = torch.flatten(out,1)
		scores = self.processOut(out)
		labels = self.classifier(scores).argmax(axis=1)
		return scores,labels

print("create model")
model = TOP_GRU()
print(model)

train_dataset = get_data(DATA_DIR_ROOT + TRAIN_DATA)
train_data_dl = DataLoader(train_dataset, batch_size = BATCH_SIZE * TIME_STEPS, drop_last=True)
test_dataset = get_data(DATA_DIR_ROOT + TEST_DATA)
test_data_dl = DataLoader(test_dataset, batch_size = BATCH_SIZE * TIME_STEPS, drop_last=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,momentum=MOMENTUM)

train_loss_list = []
test_loss_list = []
train_accuracy_trace = []
test_accuracy_trace = []
max_test_acc = 0
max_train_acc = 0

print("Start training")
for e in range(EPOCHS):
	model.train()
	train_loss = 0
	train_words_pred = []
	train_words_true = []
	for j,(feats,labels) in enumerate(train_data_dl):
		optimizer.zero_grad()
		scores,pred_labels = model(feats)
		true_labels = labels.reshape(TIME_STEPS,BATCH_SIZE)[-1,:]
		loss = criterion(scores,true_labels)
		loss.backward()
		optimizer.step()
		train_loss+=loss.item()
		train_words_true = train_words_true + true_labels.tolist()
		train_words_pred = train_words_pred + pred_labels.tolist()

	if (e>0 and e%20 == 0) or e==EPOCHS-1:
		print("epoch ",e)
		print("train_loss = ",train_loss/j)
		train_loss_list.append(train_loss/j)
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

		test_loss_list.append(test_loss/i)
		train_acc = accuracy_score(train_words_true,train_words_pred)
		test_acc = accuracy_score(test_words_true,test_words_pred)
		train_accuracy_trace.append(train_acc)
		test_accuracy_trace.append(test_acc)
		print("test_loss = ",test_loss/i)
		print("train accuracy_score = ", train_acc)
		print("test accuracy_score = ", test_acc)
		print("\n")

		if max_test_acc < test_acc:
			max_test_acc = test_acc
			torch.save(model.state_dict(),RESULTS_DIR_ROOT+args.feat+"_checkpoint_gru_"+args.type+".p")
			print("Saved Checkpoint")

PlotLoss(train_loss_list,test_loss_list,args.feat+"_loss_gru_"+args.type+".jpg")
PlotAccuracy(train_accuracy_trace,test_accuracy_trace,args.feat+"_acc_gru_"+args.type+".jpg")
PlotConfusionMatrix(train_words_true,train_words_pred,np.arange(N_CLASS),args.feat+"_conf_mat_train_gru_"+args.type+".jpg",title = "train confusion matrix")
PlotConfusionMatrix(test_words_true,test_words_pred,np.arange(N_CLASS),args.feat+"_conf_mat_test_gru_"+args.type+".jpg",title = "test confusion matrix")
