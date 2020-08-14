import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from cde_torch import *


def criterion(out, label):
    return (label - out)**2


class relu_Perceptron(torch.nn.Module):
    def __init__(self, in_layers, out_layers):
        super(relu_Perceptron, self).__init__()
        self.fc = nn.Linear(in_layers, out_layers, bias=False)
        #self.relu = torch.nn.ReLU()  # instead of Heaviside step fn

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x

   # def predict(self, x_test):


class linear_Perceptron(torch.nn.Module):
    def __init__(self, in_layers, out_layers):
        super(linear_Perceptron, self).__init__()
        self.fc = nn.Linear(in_layers, out_layers, bias=True)
        #self.relu = torch.nn.ReLU()  # instead of Heaviside step fn

    def forward(self, x):
        x = self.fc(x) #F.relu(self.fc(x))
        return x


# class ServiceNet(nn.Module):
#     def __init__(self, basis_size, marginal_beta=None):
#         super(ServiceNet, self).__init__()
#
#         self.l1 = nn.Linear(1, 2)
#         #self.l2 = nn.Linear(32, 64)
#         #self.l3 = nn.Linear(64, 32)
#         self.cde = cde_layer(2, basis_size)  # CDE layer here
#
#         # If we had some initial guess for the betas, we could include it here.
#         if marginal_beta is not None:
#             self.cde.linear.bias.data = torch.from_numpy(marginal_beta[1:]).type(torch.FloatTensor)
#
#     def forward(self, x):
#         #x = F.leaky_relu(self.l1(x))
#         #x = F.leaky_relu(self.l2(x))
#         #x = F.leaky_relu(self.l3(x))
#         beta = self.cde(x)
#         return beta

if False:
    p = Perceptron(1,1)
    print(p)

    print(list(p.parameters()))


    optimizer = optim.SGD(p.parameters(), lr=0.01, momentum=0.5)




    data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

    for epoch in range(100):
        for i, data2 in enumerate(data):
            X, Y = iter(data2)
            X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
            optimizer.zero_grad()
            outputs = p(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            if (i % 10 == 0):
                print("Epoch {} - loss: {}".format(epoch, loss.data[0]))

    print(list(p.parameters()))
#print(p(Variable(torch.Tensor([[[1]]]))))
