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

def run_epoch(epoch_no, train_dl, model, optimiser, batch_size=64, split='train') :

    for i, (inputs, targets) in enumerate(train_dl):
        # clear the gradients
        optimiser.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = cal_mpjpe(yhat, targets,avg=True)
      
        loss.requires_grad = True
        # credit assignment
        loss.backward()
        # update model weights
        optimiser.step()

    trainLoss = 0
    for i, (inputs, targets) in enumerate(train_dl):
        yhat = model(inputs)
        # calculate loss
        trainLoss += cal_mpjpe(yhat, targets,avg=True)
    trainLoss = trainLoss / len(train_dl)
    return (trainLoss,model)




if __name__=='__main__':
    dataPath = './data_train_lift.pkl'
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


    model = LiftModel(n_blocks=2, hidden_layer=1024, dropout=0.1, output_nodes=15*3)

    # Define Training Parameters 
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    optimiser = torch.optim.Adam(model.parameters(),lr=0.001)
    epochs = 12
    trainLosses = []

    # Running epochs 
    for epoch in range(epochs):
        l , model = run_epoch(epoch+1,train_dl,model,optimiser)
        trainLosses.append(l.item())
    print("Train Loss : ",trainLosses)

    modelName = './liftModel.pb'
    torch.save(model.state_dict(),modelName)
