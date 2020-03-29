import os
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader


import util
from networks import Net



#parameter initialization
parser = util.parser
args = parser.parse_args()
torch.manual_seed(args.seed)

#device selection
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
else:
    args.device = 'cpu'

#dataset split
def data_builder(args):
    dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

    return train_loader, val_loader, test_loader
   
#test function
def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)
	
#save result in txt
def save_result(test_acc, save_path):
    with open(os.path.join(save_path, 'result.txt'), 'a') as f:
        test_acc *= 100
        f.write(args.dataset+";")
        f.write("pooling_layer_type:"+args.pooling_layer_type+";")
        f.write("feature_fusion_type:"+args.feature_fusion_type+";")
        f.write(str(test_acc))
        f.write('\r\n')

#training configuration
train_loader, val_loader, test_loader = data_builder(args)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#training steps
patience = 0
min_loss = args.min_loss
for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    print("Epoch{}".format(epoch))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

#test step
model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc,test_loss = test(model,test_loader)
print("Test accuarcy:{}".format(test_acc))
save_result(test_acc, args.save_path)
