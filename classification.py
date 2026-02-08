import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        """
        初始化分类模型
        :param input_dim: 输入特征的维度（经过 PCA 和 LASSO 降维后的维度）
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出分类的类别数
        :param dropout_rate: Dropout 层的丢弃率，防止过拟合
        """
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐藏层
        self.dropout = nn.Dropout(dropout_rate)  # Dropout 层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # 隐藏层到次隐藏层
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)  # 输出层

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入数据，形状为 [batch_size, input_dim]
        :return: 模型输出，形状为 [batch_size, output_dim]
        """
        x = F.relu(self.fc1(x))  # 第一层激活函数
        x = self.dropout(x)  # Dropout 层
        x = F.relu(self.fc2(x))  # 第二层激活函数
        x = self.fc3(x)  # 输出层
        return x  # 使用 Log-Softmax 计算分类概率