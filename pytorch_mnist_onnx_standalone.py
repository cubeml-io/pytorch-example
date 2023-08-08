import torch
import torchvision

# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.dropout = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(4*4*64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(-1, 4*4*64)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

# 加载训练数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./../../dataset', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=64, shuffle=True)

if __name__ == '__main__':
    # 定义损失函数和优化器
    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # 训练模型
    model.train()
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print('Train Epoch: {}\tLoss: {:.6f}'.format(
            epoch+1, loss.item()))

    # 导出 ONNX 模型
    dummy_input = torch.randn(1, 1, 28, 28)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, "./../../model/pytorch_mnist_onnx_standalone/1/model.onnx", verbose=True, input_names=input_names, output_names=output_names)
