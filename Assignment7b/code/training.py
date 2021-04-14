# https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import numpy as np
import pickle
import torch
global input_size
global batch_size

class PoseDataset(Dataset):
	def __init__(self,data):
		self.X1 = data['joint_2d_2']
		self.X2 = data['joint_2d_1']
		self.rot = data['rot']
		self.transl = data['transl']
		self.f = data['focal_len_2']
		self.Y = data['joint_3d']
		self.camMtx = self.computeCameraMatrix(data)

	def __len__(self):
		return len(self.X1)

	def __getitem__(self,idx):
		feats = np.concatenate((self.X1[idx].flatten(),self.X2[idx].flatten(),self.rot[idx],self.transl[idx],self.f[idx]))
		return [self.X1[idx].flatten(),feats,self.Y[idx],self.camMtx]

	def computeCameraMatrix(self,data):
		fx,fy = data['focal_len_2'][0]
		k = np.identity(3)*[fx,fy,1]
		R = data['rot'][0].reshape(3,3)
		t = data['transl'][0]
		E = np.column_stack((R,t))
		P = k @ E
		return P

def resUnit(hidden_layer,dropout):
	return nn.Sequential( nn.Linear(hidden_layer,hidden_layer),
	               nn.BatchNorm1d(hidden_layer),
	               nn.ReLU(),
	               nn.Dropout(dropout))

class ResiBlock(nn.Module):
	def __init__(self,n_resUnits,hidden_layer,dropout):
		super().__init__()
		self.res_units = nn.ModuleList([resUnit(hidden_layer,dropout) for i in range(n_resUnits)])

	def forward(self,x):
		identity = x
		for unit in self.res_units:
			x = unit(x)
		return identity + x

# approach1 model
# model as given in problem description and used in 7a
class LiftModel1(nn.Module):
	def __init__(self,n_blocks,hidden_layer,dropout,output_nodes):
		super().__init__()
		self.inputLayer = nn.Linear(input_size,hidden_layer)
		self.residualLayer = nn.ModuleList()
		for i in range(n_blocks):
			block = ResiBlock(n_resUnits=2,hidden_layer=hidden_layer,dropout=dropout)
			self.residualLayer.append(block)
		self.outputLayer = nn.Linear(hidden_layer,output_nodes)

	def forward(self,x):
		x = self.inputLayer(x)
		for block in self.residualLayer:
			x = block(x)
		x = self.outputLayer(x)
		return x

# approach2 model
class LiftModel2(nn.Module):
	def __init__(self,n_blocks,hidden_layer,dropout,output_nodes):
		super().__init__()
		self.inputLayer = nn.Linear(input_size,hidden_layer)
		self.residualLayer = nn.ModuleList()
		for i in range(n_blocks):
			block = ResiBlock(n_resUnits=2,hidden_layer=hidden_layer,dropout=dropout)
			self.residualLayer.append(block)
		self.interLayer = nn.Linear(hidden_layer,output_nodes)
		self.outputLayer = nn.Linear(output_nodes,30)

	def forward(self,x):
		x = self.inputLayer(x)
		for block in self.residualLayer:
			x = block(x)
		x = self.interLayer(x)
		inter = x
		x = self.outputLayer(x)
		return inter,x

def prepare_data(path):
	datafile = open(path,'rb')
	data = pickle.load(datafile)
	dataset = PoseDataset(data)
	return dataset

def run_epoch_approach1(epoch_no, data, model, optimiser, scheduler, batch_size=64, split='train'):
	total_loss = 0.0
	count = 0
	data_dl = DataLoader(data,batch_size=batch_size,shuffle=True)
	if split == 'train':
		model.train()
		for e in range(epoch_no):
			if optimiser == 'Adam':
				optimizer = optim.Adam(model.parameters(), lr=0.01)
			if optimiser == 'SGD':
				optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
			epoch_loss = 0.0
			epoch_test_loss = 0.0
			batch_count = 0
			print("epoch:",e)
			for i,(inputs,feats,target,mtx) in enumerate(data_dl):
				optimizer.zero_grad()
				pred_3d = model(inputs)
				loss = cal_mpjpe_approach1(inputs,mtx,pred_3d)
				loss_test = test_loss(target,pred_3d)
				loss.requires_grad = True
				loss.backward()
				optimizer.step()
				epoch_loss+=loss
				epoch_test_loss+=loss_test
				batch_count+=1
			total_loss+=epoch_test_loss
			count+=batch_count
			print("epoch model loss",epoch_loss.item()/batch_count)
			print("epoch test loss",epoch_test_loss.item()/batch_count)

	if split == 'test':
		model.eval()
		for i,(inputs,feats,target,mtx) in enumerate(data_dl):
			pred_3d = model(inputs)
			loss = test_loss(target,pred_3d)
			total_loss+=loss
			count+=1
	
	avg_loss = total_loss/count
	return avg_loss.item()

def run_epoch_approach2(epoch_no, data, model, optimiser, scheduler, batch_size=64, split='train'):

	total_loss = 0.0
	count = 0
	data_dl = DataLoader(data,batch_size=batch_size,shuffle=True)
	if split == 'train':
		model.train()
		for e in range(epoch_no):
			if optimiser == 'Adam':
				optimizer = optim.Adam(model.parameters(), lr=0.01)
			if optimiser == 'SGD':
				optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
			epoch_loss = 0.0
			epoch_test_loss = 0.0
			batch_count = 0
			print("epoch:",e)
			for i,(inputs,feats,target,mtx) in enumerate(data_dl):
				optimizer.zero_grad()
				pred_3d,x = model(feats)
				loss = cal_mpjpe_approach2(inputs,x)
				loss_test = test_loss(target,pred_3d)
				loss.requires_grad = True
				loss.backward()
				optimizer.step()
				epoch_loss+=loss
				epoch_test_loss+=loss_test
				batch_count+=1
			total_loss+=epoch_test_loss
			count+=batch_count
			print("epoch model loss",epoch_loss.item()/batch_count)
			print("epoch test loss",epoch_test_loss.item()/batch_count)

	if split == 'test':
		model.eval()
		for i,(inputs,feats,target,mtx) in enumerate(data_dl):
			pred_3d,x = model(feats)
			loss = test_loss(target,pred_3d)
			total_loss+=loss
			count+=1
	
	avg_loss = total_loss/count
	return avg_loss.item()

def reconstruct(x_3d,mtx):
	n_joints = 15
	batch_size =x_3d.shape[0]
	x_ones = torch.ones((batch_size,n_joints,1))
	x_3d = torch.cat((x_3d,x_ones),axis=2)
	x_cam_ho = torch.zeros((batch_size,n_joints,3))
	x_cam = torch.zeros((batch_size,n_joints,2))
	for i in range(0,batch_size):
		x_cam_ho[i] = torch.mm(mtx[i].double(),x_3d[i].T.double()).T
		X = x_cam_ho[i,:,0]
		Y = x_cam_ho[i,:,1]
		Z = x_cam_ho[i,:,2]
		x_cam[i] = torch.FloatTensor([(X/Z).tolist(),(Y/Z).tolist()]).T
	return x_cam

def cal_mpjpe_approach1(pose_1,mtx,pred_3d,avg=True):
	pose_1 = pose_1.clone().detach()
	n_joints = 15
	batch_size = pose_1.shape[0]
	pose_1 = torch.reshape(pose_1,(batch_size, n_joints, 2))
	pred_3d = torch.reshape(pred_3d,(batch_size, n_joints, 3))
	pose_2 = reconstruct(pred_3d,mtx)
	pose_2 = pose_2.clone().detach()
	diff = pose_1-pose_2
	diff_sq = diff ** 2
	dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))
	dist_per_sample = torch.mean(dist_per_joint, axis=1)
	if avg is True:
		dist_avg = torch.mean(dist_per_sample)
	else:
		dist_avg = dist_per_sample
	return dist_avg

def cal_mpjpe_approach2(pose_1,pose_2,avg=True):
	pose_1 = pose_1.clone().detach()
	pose_2 = pose_2.clone().detach()
	n_joints = 15
	batch_size = pose_1.shape[0]
	pose_1 = torch.reshape(pose_1,(batch_size, n_joints, 2))
	pose_2 = torch.reshape(pose_2,(batch_size, n_joints, 2))
	diff = pose_1-pose_2
	diff_sq = diff ** 2
	dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))
	dist_per_sample = torch.mean(dist_per_joint, axis=1)
	if avg is True:
	    dist_avg = torch.mean(dist_per_sample)
	else:
	    dist_avg = dist_per_sample
	return dist_avg

def test_loss(pose_1, pose_2, avg=True):
    pose_1 = pose_1.clone().detach()
    pose_2 = pose_2.clone().detach()
    n_joints = 15
    batch_size = pose_1.shape[0]
    pose_1 = torch.reshape(pose_1,(batch_size, n_joints, 3))
    pose_2 = torch.reshape(pose_2,(batch_size, n_joints, 3))
    diff = pose_1-pose_2
    diff_sq = diff ** 2
    dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))
    dist_per_sample = torch.mean(dist_per_joint, axis=1)
    if avg is True:
        dist_avg = torch.mean(dist_per_sample)
    else:
        dist_avg = dist_per_sample
    return dist_avg

if __name__=="__main__":
	global input_size
	global batch_size
	input_size=15*2
	train_data = "../data/data_train.pkl"
	batch_size = 64
	n_epoch = 5
	data_dl = prepare_data(train_data)
	model1 = LiftModel1(n_blocks=2, hidden_layer=1024, dropout=0.1, output_nodes=15*3)
	loss1 = run_epoch_approach1(n_epoch,data_dl,model1,'Adam','abc',batch_size=batch_size)
	print("*"*20)
	print("APPROACH1 train loss:",loss1)
	torch.save(model1,"../results/liftModel1.pt")
	print("*"*20)
	input_size=74
	model2 = LiftModel2(n_blocks=2, hidden_layer=1024, dropout=0.05, output_nodes=15*3)
	loss2 = run_epoch_approach2(n_epoch,data_dl,model2,'Adam','abc',batch_size=batch_size)
	print("*"*20)
	print("APPROACH2 train loss:",loss2)
	torch.save(model2,"../results/liftModel2.pt")
	print("*"*20)
	if(loss1<loss2):
		torch.save(model1,"../results/bestModel.pt")
		print("Succesfully saved APPROACH1 model")
	else:
		torch.save(model2,"../results/bestModel.pt")
		print("Succesfully saved APPROACH2 model")