# -*- coding: utf-8 -*-
"""
Created on 09 16 2023 16:38:50

@Author: Yao Tianle
"""
import os
import numpy as np
import pandas as pd
import joblib
import torch

from Kd_NN_train import KD_NN


# 定义模型的测试函数
def test_model(model, test_features):
    inputs = torch.from_numpy(test_features).float()
    outputs = model(inputs)
    predictions = outputs.detach().numpy()
    kd_simu = np.ravel(predictions)
    return kd_simu


if __name__ == '__main__':
    '''---------------------------------------------------测试模型---------------------------------------------------'''
    datasets = ['NOMAD', 'BGC-Argo', 'IOCCG']

    # 创建空的表格，行为水体类型，列为算法名称
    mape_table = pd.DataFrame(index=range(1, 31), columns=['NN'])
    # 测试模型
    for dataset in datasets:
        file = './0-datasets/' + dataset + '_test_dataset.xlsx'
        data = pd.read_excel(file, keep_default_na=True)

        test_features = data.iloc[:, 0:5]
        test_labels = data.iloc[:, 5]
        test_wc = data.iloc[:, 6]

        # 加载模型(必须导入模型！！！from Kd_NN_train import KD_NN)
        kd_nn = joblib.load('Kd_NN.pkl')

        # 在测试集上评估模型
        scaler = joblib.load('scaler.pkl')
        test_features = scaler.transform(test_features)
        valid_kd_simu = test_model(kd_nn, test_features)

        # 将test_features和WC转换为DataFrame
        test_features_df = pd.DataFrame(test_features)
        test_labels_df = pd.DataFrame(test_labels)
        valid_kd_simu_df = pd.DataFrame(valid_kd_simu)
        WC_df = pd.DataFrame(test_wc)
        # 使用pd.concat()连接test_features_df和WC_df
        test_data = pd.concat([test_features_df], axis=1)
        test_data['WC_df'] = WC_df
        test_data['test_labels_df'] = test_labels_df
        test_data['valid_kd_simu_df'] = valid_kd_simu_df

        # 保存
        file_result = './0-datasets/' + dataset + '_test_dataset_result.xlsx'
        test_data.to_excel(file_result, index=False)
        print(f"保存测试集结果到{file_result}")
