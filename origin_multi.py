from torch.nn.utils.rnn import pad_sequence
import torch
import transformers
from params import MODEL_NAME_OR_PATH


class MultiModalBERTClass(torch.nn.Module):
    def __init__(self, args=None, num_labels=4, hidden_size=768, pca_feature_size=128):
        super(MultiModalBERTClass, self).__init__()
        self.args = args
        self.device = args.device
        self.bert = transformers.BertModel.from_pretrained(MODEL_NAME_OR_PATH, num_labels=num_labels)

        # CNN层
        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_norm = torch.nn.LayerNorm(hidden_size*2)

        # PCA特征处理层
        self.pca_fc = torch.nn.Linear(pca_feature_size, hidden_size // 2)
        self.pca_norm = torch.nn.LayerNorm(hidden_size // 2)
        self.pca_dropout = torch.nn.Dropout(0.5)  # 增加dropout

        cnn_output_size = hidden_size * 2  # 因为拼接了两个CNN的输出
        pca_output_size = hidden_size // 2
        combined_size = cnn_output_size + pca_output_size  # 1920

        # 修正全连接层维度
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(combined_size, hidden_size),  # 1920 -> 768
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, num_labels)  # 768 -> num_labels
        )

        self.feature_attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2 + hidden_size // 2, hidden_size),  # combined_size -> hidden_size
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 2),  # 2个权重分别对应CNN和PCA特征
            torch.nn.Softmax(dim=1)  # 确保权重和为1
        )

        # 修正残差连接维度
        self.residual = torch.nn.Linear(combined_size, num_labels)

        self.l2_reg = 0.05  # 增加L2正则化强度

    def forward(self, ids, mask, token_type_ids, pca_features, args):
        # BERT输出
        bert_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids).pooler_output

        # CNN处理
        x = bert_output.unsqueeze(-1)  # [batch_size, 768, 1]
        x = x.expand(-1, -1, 3)  # [batch_size, 768, 3]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.maxpool(x)
        x = x.mean(dim=2)  # 全局池化
        x = self.conv_norm(x)

        # PCA特征处理
        pca_x = self.pca_dropout(pca_features)
        pca_x = self.pca_fc(pca_x)
        pca_x = self.pca_norm(pca_x)

        # 特征融合
        combined = torch.cat([x, pca_x], dim=1)
        attention_weights = self.feature_attention(combined)  # [batch_size, 2]

        # 使用注意力权重加权特征
        x = x * attention_weights[:, 0].unsqueeze(1)  # CNN特征权重
        pca_x = pca_x * attention_weights[:, 1].unsqueeze(1)  # PCA特征权重

        # 特征融合
        combined = torch.cat([x, pca_x], dim=1)

        # 主路径
        output = self.fc_layers(combined)

        # 残差连接
        residual_output = self.residual(combined)

        return output + residual_output

    def get_l2_reg(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param.pow(2))
        return self.l2_reg * l2_loss