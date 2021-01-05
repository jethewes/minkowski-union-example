import torch, MinkowskiEngine as ME
from model import MobileNet

device = "cuda:0" # or whichever GPU you want!

torch.cuda.set_device(device)
data = torch.load('data/example.pt')

model = MobileNet(2, torch.nn.ReLU(), 1, 1, 4).to(device)
model.train()

test = (
    data["xfeats"].to(device),
    ME.utils.batched_coordinates([data["xcoords"]]),
    data["yfeats"].to(device),
    ME.utils.batched_coordinates([data["ycoords"]]),
)

test = model(test, device)
loss = torch.nn.modules.loss.CrossEntropyLoss()(
    test, data["truth"].reshape([1]).to(device)
)
print(loss)
loss.backward()
