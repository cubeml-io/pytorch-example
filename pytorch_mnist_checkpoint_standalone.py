import torch.optim as optim
from torchvision import datasets, transforms
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 100
num_epochs = 4
momentum = 0.5
log_interval = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def save_checkpoint(log_dir, model, optimizer, epoch):
    filepath = log_dir + '/checkpoint-resume.pt'
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


def load_checkpoint(log_dir):
    filepath = log_dir + '/checkpoint-resume.pt'
    if os.path.exists(filepath):
        return torch.load(filepath)
    return


def create_log_dir():
    log_dir = os.path.join("checkpoint/checkpoint", 'mnist')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


single_node_log_dir = create_log_dir()
print("Log directory:", single_node_log_dir)


def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader) * len(data),
                       100. * batch_idx / len(data_loader), loss.item()))


def train(train_state, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = datasets.MNIST('./../dataset', train=True, download=True,
                                   transform=transforms.Compose(
                                       [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    starting_epoch = 0
    # 加载断点的状态
    if train_state is not None:
        model.load_state_dict(train_state['model'])
        optimizer.load_state_dict(train_state['optimizer'])
        starting_epoch = train_state['epoch'] + 1

    for epoch in range(starting_epoch, num_epochs + 1):
        train_one_epoch(model, device, data_loader, optimizer, epoch)
        if epoch % 1 == 0:
            print("save checkpoint")
            save_checkpoint(single_node_log_dir, model, optimizer, epoch)


if __name__ == '__main__':
    # 加载 checkpoint 文件
    train_state = load_checkpoint(log_dir=single_node_log_dir)
    train(train_state=train_state, learning_rate=0.001)
