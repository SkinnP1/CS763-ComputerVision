from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
import numpy as np
import pickle as pkl
import torch
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("type",help="choose experiment type",choices=["SEEN","UNSEEN"],default="SEEN")
args = parser.parse_args()

DATA_DIR_ROOT = "../data/MIRACL_VC1/"

if args.type == "SEEN":
	TRAIN_DATA_DIR = "train_anno_SEEN/"
	TEST_DATA_DIR = "test_anno_SEEN/"
	TRAIN_FEAT_FILE = "train_resnet152_feats_SEEN.pkl"
	TEST_FEAT_FILE = "test_resnet152_feats_SEEN.pkl"
	FEAT_LABEL = "feats"
else:
	TRAIN_DATA_DIR = "train_anno/"
	TEST_DATA_DIR = "test_anno/"
	TRAIN_FEAT_FILE = "train_anno_with_resnet152_feats.pkl"
	TEST_FEAT_FILE = "test_anno_with_resnet152_feats.pkl"
	FEAT_LABEL = "resnet_feats"

TRAIN_DATA_FILES = [ DATA_DIR_ROOT+TRAIN_DATA_DIR+f for f in sorted(os.listdir(DATA_DIR_ROOT+TRAIN_DATA_DIR))]
TEST_DATA_FILES = [ DATA_DIR_ROOT+TEST_DATA_DIR+f for f in sorted(os.listdir(DATA_DIR_ROOT+TEST_DATA_DIR))]

class MIRACL_VC1(Dataset):
	def __init__(self,data):
		self.X = data['images']
		self.Y = data['labels']

	def __len__(self):
		return len(self.X)

	def __getitem__(self,idx):
		return [self.X[idx], self.Y[idx]]

def get_data(path):
	with open(path,'rb') as f:
		data = pkl.load(f)
	dataset = MIRACL_VC1(data)
	return dataset

class PREResNet(nn.Module):
	def __init__(self):
		super().__init__()
		# permute used because VGG accepts 3,H,W input shape
		self.resnet = models.resnet152(pretrained=True,progress=True)
		self.model = nn.Sequential(*(list(self.resnet.children())[:-1]))

	def forward(self,input):
		feats = self.model(input.permute(0,3,1,2))
		feats = torch.flatten(feats)
		return feats

model = PREResNet()
model.eval()
DATA_FILES = [TRAIN_DATA_FILES, TEST_DATA_FILES]
FEAT_FILE = [TRAIN_FEAT_FILE, TEST_FEAT_FILE]
for i in range(len(DATA_FILES)):
	data_file = DATA_FILES[i]
	feat_file = FEAT_FILE[i]
	print("Start generating feats for ",len(data_file))
	data_with_feats = {}
	data_feats = []
	data_labels = []
	for file in data_file:
		dataset = get_data(file)
		data_dl = DataLoader(dataset,batch_size = 1)
		for i,(images,labels) in enumerate(data_dl):
			feats = model(images)
			data_labels.append(labels[0].item())
			data_feats.append(feats.detach().numpy())
		print(file)

	data_with_feats[FEAT_LABEL] = data_feats
	data_with_feats["labels"] = data_labels

	with open(DATA_DIR_ROOT+feat_file,'wb') as f:
		pkl.dump(data_with_feats,f)