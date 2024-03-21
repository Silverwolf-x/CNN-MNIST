# !/usr/bin/env python3
# -*- coding: utf-8 -*-
r'''
@ author: 何雨轩
@ title: homework 2: MINST
@ description:手写数字识别--分类问题
MNIST 数据集来自美国国家标准与技术研究所National Institute of Standards and Technology (NIST).
# 60000张训练集，10000张测试集，
# 1*28*28图片

@ note:
H_out = H_in - Kernel_size + 1
H_out = H_in \ Pooling_size
使用卷积神经网络CNN，分类交叉熵Cross-Entropy做损失函数，Adam做优化
器做图片的分类，net的output是分类的数目，先用classes储存所有类别，之后用torch.max(pred,dim=1)[1]返回分类可能性最大的那个类的index

@ v0.1: 2023-04-15
自带dataset的函数：len(s)查看数据量,s.data查看x数据,s.targets查看y
回顾TensorDataset(x,y):把tensor打包为dataset
torch.FloatTensor(numpy)，把numpy转化为小数点形式的tensor

@ v0.2
UserWarning: (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:180.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s
点击跳转到该py文件，改为copy=Ture就没有这个warning了

@ v1.0 2023-04-18
修改了因为自定义split函数导致traindata拆分不均，导致准确率最高0.7的错误。现在准确率能达到0.98，回归正常水平

@ v1.1 2023-04-19
增加了绘制错误分辨图形的可视化展示
'''
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
# 防止torch包与Anaconda环境中的同一个文件出现了冲突，画不出图
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging


def same_seed(seed):
    '''固定seed保证复现'''
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        fmt=
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d-%H.%M")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    work_dir = f'./logs/{time.strftime("%Y-%m-%d",time.localtime())}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    logger_time = time.strftime("%Y-%m-%d-%H.%M", time.localtime())

    fh = logging.FileHandler(os.path.join(work_dir, f'{logger_time}.log'),
                             mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class MyModel(nn.Module):
    '''定义模型'''

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),  # (6,24,24)
            nn.MaxPool2d((2, 2)),  # (6,12,12)
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),  # (12,10,10)
            nn.MaxPool2d((2, 2))  # (12,5,5)
        )

        self.fc = nn.Sequential(nn.Linear(12 * 5 * 5, 128), nn.ReLU(),
                                nn.Linear(128, 32), nn.ReLU(),
                                nn.Linear(32, 10))
        self.net = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),  # (6,24,24)
            nn.MaxPool2d((2, 2)),  # (6,12,12)
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),  # (12,10,10)
            nn.MaxPool2d((2, 2)),  # (12,5,5)
            nn.Flatten(),
            nn.Linear(12 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(model, train_loader, criterion, optimizer, device: str, epoch: int):
    model.train()
    train_loop = tqdm(train_loader, position=0, leave=False)
    train_loss = 0
    for iter, (data, target) in enumerate(train_loop):
        data, target = data.to(device), target.to(device)
        output = model(data)
        # targets的类型是要求long(int64)，这里对齐
        loss = criterion(output, target.long())
        # 清零梯度，反向传播，更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 进度条设置
        train_loop.set_description(f'Epoch {epoch}')
        train_loop.set_postfix({'loss': loss.item()})
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def valid(model, valid_loader, criterion, device: str, epoch: int):
    model.eval()
    epoch_preds = []
    # TODO: epoch_labels累加是否等于valid_dataset？在不打乱情况下
    epoch_labels = []
    correct = 0
    valid_loop = tqdm(valid_loader,
                      position=0,
                      ncols=70,
                      leave=False,
                      desc=f'Epoch {epoch} validating')
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.long())
            test_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_preds.extend(preds.cpu().data.numpy())
            epoch_labels.extend(target.cpu().data.numpy())

    test_loss = test_loss / len(valid_loader.dataset)
    accuracy = correct / len(valid_loader.dataset)

    return test_loss, accuracy, epoch_preds, epoch_labels


def trainer(train_loader, valid_loader, model):
    #===prepare===
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    stats = {
        "best_loss": math.inf,
        "train_loss": [],
        "valid_loss": [],
        "valid_pred": None,
        "early_stop_count": 0,
    }

    for epoch in range(config.n_epoches):
        #===train mode===
        model.train()
        train_loss = train(model, train_loader, criterion, optimizer,
                           config.device, epoch)
        logger.info(f"Train | Epoch: {epoch}\t Loss: {train_loss:.2e}")
        stats['train_loss'].append(train_loss)

        #===evaluate mode===
        test_loss, accuracy, epoch_preds, epoch_labels = valid(
            model, valid_loader, criterion, config.device, epoch)

        print(f'{epoch_labels==valid_loader.dataset}1111111111')
        logger.info(
            f"Valid | Epoch: {epoch}\t Loss: {train_loss:.2e}\t Accuracy: {accuracy:.2f}\t F1 Score:{score:.2f}"
        )

        record_valid_loss.append(np.mean(loss_record))

        #===early stopping===
        if record_valid_loss[-1] < best_loss:
            best_loss = record_valid_loss[-1]
            print(
                f'Now model with loss {best_loss:4f} valid accuracy {accuracy:4f}... from epoch {epoch}'
            )
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= config.early_stop:
            print(
                f'Model is not improving for {config.early_stop} epoches. The last epoch is {epoch}.'
            )
            break
    # save_path=config.save(config.time+f'model_{loss:.3f}.ckpt')
    torch.save(model.state_dict(), config.save_model(best_loss))
    print(f'Saving model with loss {best_loss:4f}... from epoch {epoch}')
    return record_train_loss, record_valid_loss, best_loss


def loss_plot(train_loss, valid_loss):
    '''画损失图，训练误差和泛化误差'''
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(valid_loss, label='valid_loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.legend()
    plt.savefig(config.save('training loss.png'))
    plt.show()


def predict(test_data, model):
    '''注意这里载入data不是loader一批批载入
    返回pred的值，错误率，错误的坐标'''
    model.eval()
    preds = []
    incorrect_index = []
    for i, (x, y) in tqdm(enumerate(test_data), position=0, ncols=100):
        # (B, 28, 28)-->(B, 1, 28, 28)
        x = torch.unsqueeze(x, dim=1).to(config.device)
        with torch.no_grad():
            output = model(x)
            y_pred = torch.max(output, 1)[1].detach().cpu().numpy()[0]
            preds.append(y_pred)
            if (y_pred != y):
                incorrect_index.append(i)
    return preds, 1 - len(incorrect_index) / len(test_data), incorrect_index


def cm_plot(cm, accuracy):
    plt.figure()
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                linewidths=0.3,
                cmap=sns.color_palette('Blues'))
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.title(f"accuracy{accuracy:}_model's confusion matrix")
    plt.savefig(config.save('confusion matrix.png'))
    plt.show()


def incorrect_plot(test_data, preds, incorrect_index):
    '''绘制左右子图，每个图像的位置上绘制相应的标签数字'''
    num_images = len(incorrect_index)
    images = [test_data[i][0] for i in incorrect_index]

    fix_rows = 10  # 列
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(make_grid(images, nrow=fix_rows).permute(1, 2, 0))
    axs[0].set_title('True images')

    white_image = torch.ones_like(images[0]).fill_(255)
    axs[1].imshow(
        make_grid([white_image] * num_images, nrow=fix_rows).permute(1, 2, 0))
    axs[1].set_title('Predicts')
    axs[1].axis('off')
    for i in range(num_images):
        # 每个框线2像素
        axs[1].text(i % fix_rows * 30 + 16,
                    i // fix_rows * 30 + 16,
                    str(preds[i]),
                    color='black',
                    ha='center',
                    va='center')

    plt.suptitle('incorrect comparison')
    plt.savefig(config.save('incorrect comparison.png'))
    plt.show()


class config:
    '''超参数设定，用`print(pd.DataFrame([config.__dict__]))`查看当前参数'''

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 45
        self.batch_size = 512
        self.valid_ratio = 0.1
        self.folder = 'run'
        # 路径名不能出现冒号
        self.time = time.strftime(r"%Y-%m-%d_%H.%M_", time.localtime())
        #-==Important Hyperparameters===
        self.early_stop = 5
        self.learning_rate = 10e-3
        self.n_epoches = 100

    def save(self, path: str):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        return os.path.join(self.folder, self.time + path)

    def save_model(self, loss, accuracy=None):
        if accuracy == None:
            path = f'loss{loss:4f}_model.ckpt'
        else:
            path = f'accuracy{accuracy:3f}_model.ckpt'
        return self.save(path)


if __name__ == '__main__':
    config = config()
    same_seed(config.seed)
    print(f'{torch.__version__=}\n{config.device=}')
    # print(config.device)

    #===data processing===将原数据<class 'PIL.Image.Image'>转成tensor，并作标准化处理
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./',
                                train=True,
                                download=True,
                                transform=transform)
    test_data = datasets.MNIST(root='./',
                               train=False,
                               download=True,
                               transform=transform)

    n_valid = int(len(train_data) * config.valid_ratio)
    n_train = len(train_data) - n_valid
    train_dataset, valid_dataset = random_split(
        train_data, [n_train, n_valid],
        torch.Generator().manual_seed(config.seed))

    #======data processing end==
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True)

    #===training===
    model = MyModel().to(config.device)
    # print(model)
    global logger
    logger = get_logger()

    train_loss, valid_loss, best_loss = trainer(train_loader, valid_loader,
                                                model)

    #===plot loss===
    loss_plot(train_loss, valid_loss)

    #===predict===
    model = MyModel().to(config.device)
    model.load_state_dict(
        torch.load(config.save_model(best_loss), map_location=config.device))
    # 使用之前的model迁移学习
    # model.load_state_dict(torch.load(r'.\run\2023-04-18_22.38_epoch1000_score0.989000_model.ckpt',map_location=config.device),strict=False)
    preds, accuracy, incorrect_index = predict(test_data, model)
    print(f'test accuracy:{accuracy:4f}')
    os.rename(config.save_model(best_loss),
              config.save_model(best_loss, accuracy))

    #===confusion_matrix===
    cm = confusion_matrix(test_data.targets.numpy(),
                          preds,
                          labels=[i for i in range(10)])
    cm_plot(cm, accuracy)
    print(cm)

    #===incorrect comparasion===
    incorrect_plot(test_data, preds, incorrect_index)
    print('===FINISH!===')
