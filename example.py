import torch, MinkowskiEngine as ME

class CustomGlobalPool(ME.MinkowskiNetwork):
    def __init__(self):
        super(CustomGlobalPool, self).__init__(2)
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # Would need more sophisticated logic if inputs were batched
        return self.pool(x.F.transpose(0,1).unsqueeze(dim=0)).squeeze(dim=-1)

device = "cuda:0" # or whichever GPU you want!

torch.cuda.set_device(device)
data = torch.load('data/example.pt')

model1 = torch.nn.Sequential(
    ME.MinkowskiConvolution(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        stride=2,
        dimension=2,
    ),
    ME.MinkowskiGlobalPooling(),
).to(device)

model2 = torch.nn.Sequential(
    ME.MinkowskiConvolution(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        stride=2,
        dimension=2,
    ),
    CustomGlobalPool(),
).to(device)

model3 = torch.nn.Sequential(
    ME.MinkowskiChannelwiseConvolution(
        in_channels=1,
        kernel_size=2,
        stride=2,
        dimension=2,
    ),
    ME.MinkowskiGlobalPooling(),
).to(device)

model4 = torch.nn.Sequential(
    ME.MinkowskiChannelwiseConvolution(
        in_channels=1,
        kernel_size=2,
        stride=2,
        dimension=2,
    ),
    CustomGlobalPool(),
).to(device)

final = torch.nn.Sequential(
    torch.nn.Linear(
        1,
        4,
        bias=False,
    ),
).to(device)

x = ME.SparseTensor(data["xfeats"], ME.utils.batched_coordinates([data["xcoords"]]), device=device)

# First model
print("\nRunning simple model with MinkowskiConvolution and MinkowskiGlobalPooling...")
out1 = final(model1(x).F)
loss = torch.nn.modules.loss.CrossEntropyLoss()(
    out1, data["truth"].reshape([1]).to(device)
)
print("Model succeeded, loss", loss.item(), "- now backpropagating...")
loss.backward()
print("Backprop succeeded.")

# Second model
print("\nRunning simple model with MinkowskiConvolution and custom pooling...")
out2 = final(model2(x))
loss = torch.nn.modules.loss.CrossEntropyLoss()(
    out2, data["truth"].reshape([1]).to(device)
)
print("Model succeeded, loss", loss.item(), "- now backpropagating...")
loss.backward()
print("Backprop succeeded.")

# Third model
print("\nRunning simple model with MinkowskiChannelwiseConvolution and MinkowskiGlobalPooling...")
out3 = final(model3(x).F)
loss = torch.nn.modules.loss.CrossEntropyLoss()(
    out3, data["truth"].reshape([1]).to(device)
)
print("Model succeeded, loss", loss.item(), "- now backpropagating...")
loss.backward()
print("Backprop succeeded.")

# Fourth model
print("\nRunning simple model with MinkowskiChannelwiseConvolution and custom pooling...")
out4 = final(model4(x))
loss = torch.nn.modules.loss.CrossEntropyLoss()(
    out4, data["truth"].reshape([1]).to(device)
)
print("Model succeeded, loss", loss.item(), "- now backpropagating...")
loss.backward()
print("Backprop succeeded.")

