import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import VAENet
from loss import KLDivLoss

EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SZ = 16

transform = transforms.Compose(
    [transforms.ToTensor(),])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SZ,
                                          shuffle=True, num_workers=2)

net = VAENet()
mse = nn.MSELoss()
kl_div = KLDivLoss()
optimizer = optim.NAdam(net.parameters(), lr=0.001) # weight decay is extremely harmful

net.to(device)
for epoch in range(EPOCHS):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data

        inputs = inputs.to(device).flatten(-2)
        optimizer.zero_grad()

        outputs, m, gamma = net(inputs)

        mse_loss = mse(outputs, inputs)
        kl_div_loss = kl_div(m, gamma)
        loss = mse_loss + kl_div_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
torch.save(net.state_dict(), "model.pth")
print('Finished Training')
