import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def criterion(out, label):
    return (label - out)**2


class Perceptron(torch.nn.Module):
    def __init__(self, in_layers, out_layers):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(in_layers, out_layers, bias=False)
        #self.relu = torch.nn.ReLU()  # instead of Heaviside step fn

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x

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
