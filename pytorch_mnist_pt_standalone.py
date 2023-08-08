import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


# 下载数据
def download_dataset(data_dir, batch_size):
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN()

# 训练代码
def train(writer, model, optimizer, num_epochs, train_data, test_data):
    total_step = len(train_data)
    criterion = nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data):
            # 前向传播和计算损失
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每 100 步记录一次训练准确率
            if (i + 1) % 100 == 0:
                # 记录训练损失
                x_index = epoch * total_step + i
                writer.add_scalar('Train/Loss', loss.item(), x_index)
                correct = 0
                total = 0
                for images, labels in test_data:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                # 记录训练准确率
                writer.add_scalar('Train/Accuracy', accuracy, x_index)
                test_loss, test_accuracy = test(model, test_data, criterion)
                # 记录测试损失和准确率
                writer.add_scalar('Test/Loss', test_loss, x_index)
                writer.add_scalar('Test/Accuracy', test_accuracy, x_index)
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), test_loss, test_accuracy))

                # 记录权重和梯度的直方图
                for name, param in model.named_parameters():
                    writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), x_index)
                    writer.add_histogram(name + '/weight', param.data.cpu().numpy(), x_index)
        # 记录每个epoch的一些图像
        images = next(iter(test_data))[0][:10]
        grid = make_grid(images)
        writer.add_image('Test/Images', grid, epoch)
    return model


# 记录测试损失和准确率
def test(model, test_data, criterion):
    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        test_total = 0
        for images, labels in test_data:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_data)
        test_accuracy = 100 * test_correct / test_total
        return test_loss, test_accuracy


def setup_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default="./../../dataset")            # /data/dataset/数据名称
    parser.add_argument('--output-dir', default="./../../model/pytorch_mnist_pt_standalone/1")  # 模型输出
    parser.add_argument('--tensorboard-dir', default="./../../tensorboard/pytorch_mnist_pt_standalone")
    parser.add_argument('--batch-size', default=16)
    parser.add_argument('--num-epochs', default=1)
    parser.add_argument('--learning-rate', default=0.01)
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = setup_config()
    print("train start")
    # 开启tensorboard
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    # 下载数据
    train_data, test_data = download_dataset(data_dir=args.data_dir, batch_size=args.batch_size)
    # 定义模型和opt
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 将模型架构写入TensorBoard
    images, labels = next(iter(train_data))
    writer.add_graph(model, images)
    # 开始训练
    model = train(writer, model, optimizer, args.num_epochs, train_data, test_data)
    # 保存模型两种格式
    traced_model = torch.jit.trace(model, torch.randn(1, 1, 28, 28))
    model_dir = args.output_dir + "/model.pt"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.jit.save(traced_model, model_dir)
    # 保存成onnx模型
    # torch.onnx.export(model, torch.randn(1, 1, 28, 28), "model.onnx", verbose=True)

    # torch.save(model, model_dir)
    # 这样保存模型会有问题，推理的错误信息如下：
    # Internal: failed to load model 'pytorch_mnist_pt_standalone': PytorchStreamReader failed locating file constants.pkl: file not found
    print("train success")
    writer.close()

# python /data/code/train-dev-demo/train.py --data-dir="/data/dataset-dev-demo" --output-dir="/data/model" --tensorboard-dir="/data/tensorboard"