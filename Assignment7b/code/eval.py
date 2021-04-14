# https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

from training import *

if __name__=="__main__":
	test_data = "../data/data_train.pkl"
	batch_size = 64
	data = prepare_data(test_data)
	model1 = torch.load("../results/liftModel1.pt")
	# print(model1)
	loss = run_epoch_approach1(1,data,model1,'Adam','abc',batch_size=batch_size,split='test')
	print("APPROACH1 test loss:",loss)
	model2 = torch.load("../results/liftModel2.pt")
	# print(model2)
	loss = run_epoch_approach2(1,data,model2,'Adam','abc',batch_size=batch_size,split='test')
	print("APPROACH2 test loss:",loss)