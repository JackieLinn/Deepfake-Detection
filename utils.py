import os
import sys
import random
import logging

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from models.alexnet import AlexNet
from models.googlenet import GoogLeNet
from models.resnet import resnet101
from models.resnext import resnext101_32x8d as resnext101
from models.densenet import densenet201
from models.swintransformer import swin_small
from models.mobilenet import MobileNetV2
from models.mobilevit import mobile_vit_small as mit
from models.regnet import create_regnet
from models.convnext import convnext_small
from models.efficientnet import efficientnet_b2 as efficientnet
from models.efficientnetv2 import efficientnetv2_m as efficientnetv2
from models.xception import xception


def set_random_seed(seed):
    random.seed(seed)  # Python的随机种子
    np.random.seed(seed)  # Numpy的随机种子
    torch.manual_seed(seed)  # PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # PyTorch的GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU，设置所有GPU的种子
    # torch.backends.cudnn.deterministic = True  # 确保卷积操作是确定的
    # torch.backends.cudnn.benchmark = False  # 禁止cudnn自动优化，确保每次运行的一致性


# 读取数据集
def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # train 和 test 文件夹路径
    train_folder = os.path.join(root, 'train')
    test_folder = os.path.join(root, 'test')

    # 存储图片路径和对应标签
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应标签
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = []  # 存储测试集图片对应标签

    # 支持的文件后缀类型
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    # 读取 train 文件夹的所有图片
    for img_name in os.listdir(train_folder):
        img_path = os.path.join(train_folder, img_name)
        if os.path.splitext(img_name)[-1] in supported:
            label = int(img_name.split('_')[0])  # 通过图片名称前缀来获取标签，0 或 1
            train_images_path.append(img_path)
            train_images_label.append(label)

    # 读取 test 文件夹的所有图片
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        if os.path.splitext(img_name)[-1] in supported:
            label = int(img_name.split('_')[0])  # 通过图片名称前缀来获取标签，0 或 1
            test_images_path.append(img_path)
            test_images_label.append(label)

    # 打印数据集统计信息
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(test_images_path)} images for testing.")

    # 确保训练集和测试集都不为空
    assert len(train_images_path) > 0, "Number of training images must be greater than 0."
    assert len(test_images_path) > 0, "Number of testing images must be greater than 0."

    # 返回训练集和测试集路径及标签
    return train_images_path, train_images_label, test_images_path, test_images_label


# 一次训练函数
def train(model, optimizer, data_loader, device, epoch):
    model.train()  # 模型设定为训练模式
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # 对标签进行平滑处理
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0  # 样本计数器
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  # 计算正确预测的样本数

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()  # 累计损失

        train_loss = accu_loss.item() / (step + 1)
        train_acc = accu_num.item() / sample_num

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1, train_loss, train_acc)

        if not torch.isfinite(loss):  # 检查损失值是否有限，如果是无穷大或者NAN则终止
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return train_loss, train_acc


# 评估函数，评估测试集
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()  # 模型设定为评估模式

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    # 用于保存所有的真实标签和预测结果
    all_labels = []
    all_preds = []
    all_preds_proba = []  # 存储预测的概率，用于计算AUC

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # 前向传播
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 计算损失
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 收集所有的预测和标签
        all_labels.extend(labels.cpu().numpy())  # 将数据从 GPU 转移到 CPU
        all_preds.extend(pred_classes.cpu().numpy())
        all_preds_proba.extend(torch.softmax(pred, dim=1).cpu().numpy())  # softmax得到类别概率

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    # 计算精确率、召回率、F1 分数和 ROC-AUC
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, np.array(all_preds_proba)[:, 1])

    return (accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            precision, recall, f1, auc, all_labels, all_preds)


# 日志函数
def get_logger(model_name: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(f"./logs/{model_name}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# 一整次epoch的循环
def train_loop(model, optimizer, train_loader, test_loader, device, epoch, logger):
    # train
    train_loss, train_acc = train(model=model,
                                  optimizer=optimizer,
                                  data_loader=train_loader,
                                  device=device,
                                  epoch=epoch)
    # validate
    test_loss, test_acc, precision, recall, f1, auc, all_labels, all_preds = evaluate(model=model,
                                                                                      data_loader=test_loader,
                                                                                      device=device,
                                                                                      epoch=epoch)

    if logger is not None:
        logger.info(f'Epoch: {epoch + 1}, '
                    f'Training loss: {train_loss}, '
                    f'Training accuracy: {train_acc}, '
                    f'Validating loss: {test_loss}, '
                    f'Validating accuracy: {test_acc}, '
                    f'Precision: {precision}, '
                    f'Recall: {recall}, '
                    f'f1 score: {f1}, '
                    f'auc: {auc}')

    return test_acc, all_labels, all_preds


def create_model(model_name: str, device, num_classes: int):
    if model_name == "alexnet":
        model = AlexNet(num_classes=num_classes).to(device)
    elif model_name == 'googlenet':
        model = GoogLeNet(num_classes=num_classes).to(device)
    elif model_name == 'resnet':
        model = resnet101(num_classes=num_classes).to(device)
    elif model_name == 'resnext':
        model = resnext101(num_classes=num_classes).to(device)
    elif model_name == 'densenet':
        model = densenet201(num_classes=num_classes).to(device)
    elif model_name == 'swint':
        model = swin_small(num_classes=num_classes).to(device)
    elif model_name == 'mobilenet':
        model = MobileNetV2(num_classes=num_classes).to(device)
    elif model_name == 'mit':
        model = mit(num_classes=num_classes).to(device)
    elif model_name == 'regnet':
        model = create_regnet(num_classes=num_classes).to(device)
    elif model_name == 'convnext':
        model = convnext_small(num_classes=num_classes).to(device)
    elif model_name == 'efficientnet':
        model = efficientnet(num_classes=num_classes).to(device)
    elif model_name == 'ev2':
        model = efficientnetv2(num_classes=num_classes).to(device)
    elif model_name == 'xception':
        model = xception(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Please select an existing model.")

    return model
