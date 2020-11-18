import torch, MinkowskiEngine as ME
from model import MobileNet

torch.cuda.set_device("cuda:2")
data = torch.load('data/example.pt')

model = MobileNet(2, torch.nn.ReLU(), 1, 1, 4).to("cuda:2")
model.train()

test = ( data['xfeats'].to("cuda:2"),
         ME.utils.batched_coordinates([data['xcoords']]),
         data['yfeats'].to("cuda:2"),
         ME.utils.batched_coordinates([data['ycoords']]))

test = model(test)
loss = torch.nn.modules.loss.CrossEntropyLoss()(test, data['truth'].reshape([1]).to(test.device))
print(loss)
loss.backward()

