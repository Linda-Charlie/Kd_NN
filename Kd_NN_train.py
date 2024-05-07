# -*- coding: utf-8 -*-
"""
Created on 09 01 2023 16:11:15

@Author: Yao Tianle
"""
import os
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from A_0_OC_3S import OC_3S_v1


class KD_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(KD_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 在最后一层使用Softplus激活函数代替ReLU
        x = F.softplus(self.fc3(x))
        return x


# 将数据集按照水体类型WC分组
def group_data_by_WC(Rrs, kd_true, WC):
    grouped_data = {}
    unique_WC = np.unique(WC)
    for wc in unique_WC:
        indices = np.where(WC == wc)[0]
        grouped_data[wc] = (Rrs.iloc[indices], kd_true.iloc[indices])
    return grouped_data


# 对每个水体类型的数据进行随机抽样，划分为训练集和测试集
def split_dataset_by_WC(grouped_data, test_size, random_state=42, min_samples=1):
    train_features, train_labels, train_wc, test_features, test_labels, test_wc = [], [], [], [], [], []

    for wc, (features, labels) in grouped_data.items():
        if len(features) <= min_samples:
            train_feats, train_lbls, train_wct = features, labels, [wc] * len(features)
            test_feats, test_lbls, test_wct = np.empty((0, 5)), [], []  # 空数组的维度与其他数组一致
        else:
            train_feats, test_feats, train_lbls, test_lbls, train_wct, test_wct = train_test_split(
                features, labels, [wc] * len(features), test_size=test_size, random_state=random_state
            )
        train_features.append(train_feats)
        train_labels.append(train_lbls)
        train_wc.extend(train_wct)
        test_features.append(test_feats)
        test_labels.append(test_lbls)
        test_wc.extend(test_wct)

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)
    test_features = np.concatenate(test_features)
    test_labels = np.concatenate(test_labels)

    return train_features, train_labels, train_wc, test_features, test_labels, test_wc


# 定义模型的训练函数
def train_model(model, train_features, train_labels, num_epochs, learning_rate, writer, dataset_n):
    train_labels = train_labels.to_numpy()

    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

    for epoch in range(num_epochs):
        inputs = torch.from_numpy(train_features).float()
        targets = torch.from_numpy(train_labels).float()
        targets = targets.unsqueeze(1)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # 反向传播、更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 计算并写入训练集上的损失值
        writer.add_scalar(dataset_n + '_Train Loss', loss.item(), epoch)


if __name__ == '__main__':
    '''---------------------------------------------------生成训练集---------------------------------------------------'''
    datasets = ['NOMAD', 'BGC-Argo', 'IOCCG']

    # 创建空列表来存储所有数据集的训练集
    train_features_all = []
    train_labels_all = []
    train_WC_all = []

    # 读取和处理每个数据集
    for i in range(len(datasets)):
        file = os.path.join('./0-datasets', datasets[i] + '.csv')
        data = pd.read_csv(file, keep_default_na=True)

        Rrs = np.array(data[['B1', 'B2', 'B3', 'B4', 'B5']])
        test_bands = np.array([412, 443, 488, 555, 667], 'int32')
        [WC, Score] = OC_3S_v1(Rrs, test_bands)  # WC: water class
        data["WC"] = WC
        data["Score"] = Score
        # 筛选出满足 Score >= 0.5 条件的行
        high_score_indices = np.where(Score >= 0.5)[0]
        data_u = data.iloc[high_score_indices, :]
        # 如果在subset指定的列中的任何一个列有缺失值（NaN），则删除相应的行
        data_u = data_u.dropna(subset=['B1', 'B2', 'B3', 'B4', 'B5', 'Kd_490'], how='any')
        # 剔除Kd小于等于0的行
        data_u = data_u[data_u['Kd_490'] > 0]

        Rrs_u = data_u.iloc[:, 4:9]
        kd_true = data_u.iloc[:, 9]
        WC_u = data_u['WC']

        # 按照水体类型WC分组数据
        grouped_data = group_data_by_WC(Rrs_u, kd_true, WC_u)

        # 对每个水体类型的数据进行随机抽样，划分为训练集和测试集
        test_size = 0.2
        train_features, train_labels, train_wc, test_features, test_labels, test_wc = split_dataset_by_WC(grouped_data,
                                                                                                          test_size)

        # 将当前数据集的训练集添加到总训练集列表中
        train_features_all.append(train_features)
        train_labels_all.append(train_labels)
        train_WC_all.append(train_wc)

        # 对于测试集：创建DataFrame对象，保存DataFrame为Excel文件
        df = pd.DataFrame(test_features, columns=['B1', 'B2', 'B3', 'B4', 'B5'])
        df['kd490'] = test_labels
        df['WC'] = test_wc
        excel_file = './0-datasets/' + datasets[i] + '_test_dataset.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"保存{datasets[i]}测试集到{excel_file}")

    # 将所有数据集的训练集合并
    train_features_all = np.concatenate(train_features_all, axis=0)
    train_labels_all = np.concatenate(train_labels_all, axis=0)
    train_WC_all = np.concatenate(train_WC_all, axis=0)
    # 保存
    df_train = pd.DataFrame(train_features_all, columns=['B1', 'B2', 'B3', 'B4', 'B5'])
    df_train['kd490'] = train_labels_all
    df_train['WC'] = train_WC_all  # 添加水体类型列
    excel_file_train = './0-datasets/train_dataset.xlsx'
    df_train.to_excel(excel_file_train, index=False)
    print(f"保存训练集到{excel_file_train}")

    '''---------------------------------------------------训练模型---------------------------------------------------'''

    # 训练集文件的路径列表
    files = ['0-datasets/train_dataset.xlsx']
    # 读取每个文件的前六列并存储到列表中
    dfs = [pd.read_excel(file, usecols=range(6)) for file in files]
    # 合并所有DataFrame为一个
    train_dataset = pd.concat(dfs, ignore_index=True)
    # 如果需要，可以将合并后的DataFrame保存为新的Excel文件
    train_dataset.to_excel('train_dataset_combined.xlsx', index=False)

    # 提取前五列作为训练特征
    train_features_all = train_dataset.iloc[:, :5]
    # 提取第六列作为训练标签
    train_labels_all = train_dataset.iloc[:, 5]

    log_dir = 'show'
    writer = SummaryWriter(log_dir)

    # 标准化合并后的训练集
    scaler = StandardScaler()
    train_features_all = scaler.fit_transform(train_features_all)
    # 保存标准化器到磁盘
    joblib.dump(scaler, 'scaler.pkl')

    # 创建模型实例
    input_size = train_features_all.shape[1]
    output_size = 1  # 根据输出数据的维度调整
    kd_nn = KD_NN(input_size, output_size)

    # 记录训练开始时间
    start_time = time.time()

    # 训练模型
    num_epochs = 2000
    learning_rate = 0.005
    train_model(kd_nn, train_features_all, train_labels_all, num_epochs, learning_rate, writer, 'all')

    # 记录训练结束时间并计算总训练时长
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f} seconds')

    # 保存模型到文件
    joblib.dump(kd_nn, 'Kd_NN.pkl')

    writer.close()
