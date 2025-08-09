import numpy as np
import os
import pandas as pd
import torch


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    
    trainX1 = np.array(train_X)
    trainY1 = np.array(train_Y)
    testX1 = np.array(test_X)
    testY1 = np.array(test_Y)
    print(trainX1.shape)
    print(trainY1.shape)
    print(testX1.shape)
    print(testY1.shape)
    
    return trainX1, trainY1, testX1, testY1

def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset

def generate_SceneGCN_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val

    data1 = np.mat(data)
    train_size = int(time_len * split_ratio)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]

    poi_data = {}
    poi_dir = r"C:\DCLASS\myJupyter\paper2\application\prediction\model\data\POI"
    for filename in os.listdir(poi_dir):
        if filename.endswith(".csv"):
            name = os.path.splitext(filename)[0]
            file_path = os.path.join(poi_dir, filename)
            data = pd.read_csv(file_path, header=None).fillna(0).values
            data_max = np.max(data)
            data_nor = data if data_max == 0 else data / data_max
            poi_data[name] = data_nor

    # 创建用于存储所有归一化降水数据的字典
    precip_data = {}
    # 设置目录路径
    precip_dir = r"C:\DCLASS\myJupyter\paper2\application\prediction\model\data\precipitation"
    # 遍历文件夹中的所有 CSV 文件
    for filename in os.listdir(precip_dir):
        if filename.endswith(".csv"):
            name = os.path.splitext(filename)[0]  # 去除文件后缀 .csv，作为字典的 key
            file_path = os.path.join(precip_dir, filename)
            # 读取 CSV 内容（无表头）
            data = pd.read_csv(file_path, header=None).fillna(0).values  # 转为 numpy 数组，填充空值为 0
            # 归一化：最大值归一（防止除以 0）
            data_max = np.max(data)
            data_nor = data if data_max == 0 else data / data_max
            # 存入字典
            precip_data[name] = data_nor

    # === 构造训练集样本，嵌入所有 POI 的第一行 ===
    trainX, trainY = [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a1 = train_data[i: i + seq_len + pre_len]
        seq_input = a1[0:seq_len]  # shape: (seq_len, feature_dim)

        # 降水拼接（每个降水源都按行堆叠）
        precip_stack = np.row_stack([
            np.tile(precip_data[name][i: i + seq_len], (1, seq_input.shape[1]))  # ← 横向复制
            for name in precip_data
        ])  # shape: (seq_len * num_precip, num_nodes)

        # POI拼接（取每个POI的第0行）
        poi_stack = np.row_stack([
            poi_data[name][0:1] for name in poi_data
        ])  # shape: (n_poi, num_nodes) 或 (n_poi, ?)

        # 全拼接
        full_input = np.row_stack([seq_input, precip_stack, poi_stack])

        trainX.append(full_input)
        trainY.append(a1[seq_len: seq_len + pre_len])


    testX, testY = [], []
    for i in range(len(test_data) - seq_len - pre_len):
        a1 = test_data[i: i + seq_len + pre_len]
        seq_input = a1[0:seq_len]  # shape: (seq_len, feature_dim)

        # 降水拼接（每个降水源都按行堆叠）
        precip_stack = np.row_stack([
            np.tile(precip_data[name][i: i + seq_len], (1, seq_input.shape[1]))  # ← 横向复制
            for name in precip_data
        ])  # shape: (seq_len * num_precip, num_nodes)

        # POI拼接（取每个POI的第0行）
        poi_stack = np.row_stack([
            poi_data[name][0:1] for name in poi_data
        ])  # shape: (n_poi, num_nodes) 或 (n_poi, ?)

        # 全拼接
        full_input = np.row_stack([seq_input, precip_stack, poi_stack])

        testX.append(full_input)
        testY.append(a1[seq_len: seq_len + pre_len])
    
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    print(trainX1.shape)
    print(trainY1.shape)
    print(testX1.shape)
    print(testY1.shape)
    
    return trainX1, trainY1, testX1, testY1

def generate_SceneGCN_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_SceneGCN_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset