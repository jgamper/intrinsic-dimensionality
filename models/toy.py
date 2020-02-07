import torch
import torch.nn.functional as F

class RegularCNNModel(torch.nn.Module):
    def __init__(self):
        super(RegularCNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3)
        # self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(32 * 5 * 5, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # print(x.size())

        x = x.view(-1, 32 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class FCNAsInPAper(torch.nn.Module):
    def __init__(self):
        super(FCNAsInPAper, self).__init__()
        self.fc1 = torch.nn.Linear(784, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)