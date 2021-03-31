import torch
import numpy as np 
import pickle

class ResidualBlock(torch.nn.Module):

    def __init__(self,input_size,hidden_layer,dropout):
        super(ResidualBlock, self).__init__()
        self.layer1 = torch.nn.Linear(input_size,hidden_layer)
        self.batch1 = torch.nn.BatchNorm1d(hidden_layer)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.drop1 = torch.nn.Dropout(p=dropout)
        self.layer2 = torch.nn.Linear(hidden_layer,hidden_layer)
        self.batch2 = torch.nn.BatchNorm1d(hidden_layer)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.drop2 = torch.nn.Dropout(p=dropout)

    def forward(self,X):
        residual =  X
        out = self.layer1(X)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.layer2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out += residual
        return (out)


class LiftModel(torch.nn.Module):
    def __init__(self, n_blocks=2, hidden_layer=1024, dropout=0.1, output_nodes=15*3):
        super(LiftModel, self).__init__()
        self.input = torch.nn.Linear(30,hidden_layer)
        self.l1 = ResidualBlock(hidden_layer,hidden_layer,dropout)
        self.l2 = ResidualBlock(hidden_layer,hidden_layer,dropout)
        self.final = torch.nn.Linear(hidden_layer,output_nodes)

    def forward(self, X):
        out = self.input(X)
        out = self.l1(out)
        out = self.l2(out)
        out = self.final(out)
        return (out)

def cal_mpjpe(p1, p2,avg):
    pose_1 = p1.clone().detach()
    pose_2 = p2.clone().detach()
    n_joints = 15
    batch_size = pose_1.shape[0]
    pose_1 = torch.reshape(pose_1,(batch_size,n_joints,3))
    pose_2 = torch.reshape(pose_2,(batch_size,n_joints,3))
    diff = pose_1-pose_2
    diff_sq = diff ** 2 
    dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))
    dist_per_sample = torch.mean(dist_per_joint, axis=1)
    if avg is True:
        dist_avg = torch.mean(dist_per_sample)
    else:
        dist_avg = dist_per_sample
    return dist_avg




if __name__=='__main__':
    dataPath = './data_test_lift.pkl'
    modelName = './liftModel.pb'
    with open(dataPath, 'rb') as f:
        data = pickle.load(f)

    X = []
    for i in data['joint_2d_1']:
        a = i.flatten()
        X.append(a)

    X = np.array(X,dtype=np.float32)

    Y = []
    for i in data['joint_3d']:
        a = i.flatten()
        Y.append(a)

    Y = np.array(Y,dtype=np.float32)

    dataset = []
    for i in range(len(X)):
        dataset.append([X[i],Y[i]])

    test_dl = torch.utils.data.DataLoader(dataset, batch_size=64,  shuffle=False)

    model = LiftModel()
    model.load_state_dict(torch.load(modelName))

    testLoss = 0
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        # calculate loss
        testLoss += cal_mpjpe(yhat, targets,avg=True)
    testLoss = testLoss / len(test_dl)
    print("Test Loss : " , testLoss.item())