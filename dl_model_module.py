"""
FTTRANSFORMER多因子特征提取模块

本模块使用PyTorch原生实现的FTTransformer (Feature Tokenizer Transformer)模型
从多因子数据中提取深度学习特征。

主要功能：
1. 使用Transformer架构学习因子的非线性组合
2. 从CLS token提取全局特征表示
3. 自动支持CPU/GPU加速

调参指南：
- d_model: 嵌入维度，越大模型越复杂，默认为64
- n_heads: 注意力头数，需要能被d_model整除
- num_layers: Transformer层数，越深表达能力越强但训练越慢
- epochs: 训练轮数，需要根据数据量调整
- batch_size: 批大小，越大训练越快但显存占用越高
- lr: 学习率，建议1e-4到1e-2之间
- dropout: dropout比例，防止过拟合

使用示例：
    from dl_model_module import add_fttransformer_features_simple

    train_data_new, test_data_new, model = add_fttransformer_features_simple(
        train_data, test_data, factor_list, label,
        epochs=10, batch_size=256
    )
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
import warnings
warnings.filterwarnings('ignore')


class TabularDataset(Dataset):
    """PyTorch数据集类，用于包装特征数据"""

    def __init__(self, X_num):
        self.X_num = torch.FloatTensor(X_num)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return self.X_num[idx]


class FTTransformer(nn.Module):
    """
    FTTransformer模型 (Feature Tokenizer Transformer)

    架构说明：
    1. 输入投影层：将每个特征映射到d_model维空间
    2. CLS Token：用于聚合全局信息
    3. 位置编码：添加位置信息
    4. Transformer编码器：学习特征交互
    5. 输出层：从CLS token提取特征

    参数：
        num_features: 输入特征数量
        d_model: 嵌入维度 (默认64)
        n_heads: 注意力头数 (默认4)
        num_layers: Transformer层数 (默认2)
        d_ff: 前馈网络维度 (默认128)
        dropout: Dropout比例 (默认0.1)
    """

    def __init__(self, num_features, d_model=64, n_heads=4, num_layers=2, d_ff=128, dropout=0.1):
        super(FTTransformer, self).__init__()

        self.num_features = num_features
        self.d_model = d_model

        # 输入投影层：将每个特征映射到d_model维空间
        self.input_projection = nn.Linear(1, d_model)

        # CLS Token：用于聚合全局信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置编码：添加位置信息
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features + 1, d_model))

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入特征，形状为 (batch_size, num_features)

        返回：
            enhanced_features: 提取的特征，形状为 (batch_size, d_model)
        """
        batch_size = x.size(0)

        # 将每个特征映射到d_model维空间
        x = x.unsqueeze(-1)  # (batch, num_features) -> (batch, num_features, 1)
        x = self.input_projection(x)  # (batch, num_features, d_model)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, x], dim=1)  # (batch, num_features+1, d_model)

        # 添加位置编码
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]

        # Transformer编码
        output = self.transformer(tokens)

        # 提取CLS token的输出作为全局特征
        cls_output = output[:, 0, :]

        # 通过全连接层
        enhanced_features = self.fc(cls_output)

        return enhanced_features


def prepare_data(train_data, factor_list, label_col=None):
    """
    数据预处理函数

    参数：
        train_data: 训练数据DataFrame
        factor_list: 特征列名列表
        label_col: 标签列名（可选）

    返回：
        X_scaled: 标准化后的特征矩阵
        scaler: StandardScaler对象
        available_features: 可用的特征列表
    """
    # 获取可用的特征列
    available_features = [col for col in factor_list if col in train_data.columns]

    # 如果指定了标签列，从特征列表中移除
    if label_col and label_col in available_features:
        available_features.remove(label_col)

    # 提取特征数据
    X = train_data[available_features].copy()

    # 将对象类型列转换为数值
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 填充缺失值
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, available_features


def extract_fttransformer_features(train_data, factor_list, label_col=None,
                                   d_model=64, n_heads=4, num_layers=2,
                                   epochs=20, batch_size=256, lr=0.001,
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   trained_model=None, scaler=None):
    """
    提取FTTransformer特征

    参数：
        train_data: 训练数据DataFrame
        factor_list: 特征列名列表
        label_col: 标签列名（可选）
        d_model: 嵌入维度 (默认64，越大模型越复杂)
        n_heads: 注意力头数 (默认4，需要能被d_model整除)
        num_layers: Transformer层数 (默认2，越深表达能力越强)
        epochs: 训练轮数 (默认20，数据量大时需要更多轮)，如果传入trained_model则设为0
        batch_size: 批大小 (默认256，越大训练越快但显存占用越高)
        lr: 学习率 (默认0.001，建议1e-4到1e-2之间)
        device: 设备 (自动检测CUDA)
        trained_model: 已训练好的模型，如果为None则训练新模型
        scaler: 已训练好的scaler，如果为None则训练新scaler

    返回：
        features_df: 提取的特征DataFrame
        model: 训练好的模型
        scaler: 数据标准化器
    """
    logging.info(f"FTTRANSFORMER模块：开始提取特征，设备: {device}")

    # 数据预处理
    X_scaled, scaler, available_features = prepare_data(train_data, factor_list, label_col)

    logging.info(f"FTTRANSFORMER模块：特征数量: {len(available_features)}")

    # 如果传入了已训练的模型，直接使用，否则训练新模型
    if trained_model is not None and epochs == 0:
        model = trained_model
        logging.info("FTTRANSFORMER模块：使用已训练的模型提取特征")
    else:
        # 创建数据加载器
        dataset = TabularDataset(X_scaled)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        model = FTTransformer(
            num_features=len(available_features),
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            d_ff=d_model * 2,
            dropout=0.1
        ).to(device)

        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # 训练模型
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(device)

                optimizer.zero_grad()

                # 前向传播
                enhanced_features = model(batch)

                # 自编码器重建损失
                reconstruction = torch.mean(enhanced_features, dim=1, keepdim=True)
                reconstruction = reconstruction.repeat(1, enhanced_features.size(1))
                loss = criterion(enhanced_features, reconstruction)

                # 反向传播
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                logging.info(f"FTTRANSFORMER模块：Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    # 提取特征
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch_end = min(i + batch_size, len(X_scaled))
            batch_x = torch.FloatTensor(X_scaled[i:batch_end]).to(device)

            enhanced = model(batch_x)
            all_features.append(enhanced.cpu().numpy())

    features = np.vstack(all_features)

    # 创建特征DataFrame
    feature_cols = [f'ftt_feature_{i}' for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_cols, index=train_data.index)

    logging.info(f"FTTRANSFORMER模块：特征提取完成，新特征维度: {features_df.shape}")

    return features_df, model, scaler


def add_fttransformer_features(train_data, test_data, factor_list, label_col=None,
                                epochs=20, batch_size=512, d_model=64,
                                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    添加FTTransformer特征到训练集和测试集

    参数：
        train_data: 训练数据DataFrame
        test_data: 测试数据DataFrame
        factor_list: 特征列名列表
        label_col: 标签列名（可选）
        epochs: 训练轮数 (默认20)
        batch_size: 批大小 (默认512)
        d_model: 嵌入维度 (默认64)
        device: 设备 (自动检测CUDA)

    返回：
        train_result: 只包含新特征的训练数据
        test_result: 只包含新特征的测试数据
        model: 训练好的模型

    调参建议：
        - 数据量小( <10万): d_model=32, epochs=10, batch_size=128
        - 数据量中等(10-100万): d_model=64, epochs=20, batch_size=256
        - 数据量大( >100万): d_model=128, epochs=30, batch_size=512
    """
    logging.info(f"FTTRANSFORMER模块：开始处理，设备: {device}")

    # 【重要】只用训练集训练模型，避免数据泄露
    train_features_df, model, scaler = extract_fttransformer_features(
        train_data, factor_list, label_col,
        d_model=d_model, epochs=epochs, batch_size=batch_size,
        device=device
    )

    # 用训练好的模型对测试集提取特征
    test_features_df, _, _ = extract_fttransformer_features(
        test_data, factor_list, label_col,
        d_model=d_model, epochs=0, batch_size=batch_size,
        device=device,
        trained_model=model,
        scaler=scaler
    )

    # 只保留新生成的特征列（ftt_feature_*），不要原始特征
    ftt_cols = [col for col in train_features_df.columns if col.startswith('ftt_')]
    train_result = train_features_df[ftt_cols]
    test_result = test_features_df[ftt_cols]

    # 恢复原始索引
    if isinstance(train_data.index, pd.MultiIndex):
        train_result = train_result.set_index(train_data.index)
        test_result = test_result.set_index(test_data.index)

    logging.info(f"FTTRANSFORMER模块：特征添加完成，训练集: {train_result.shape}, 测试集: {test_result.shape}")

    return train_result, test_result, model


def add_fttransformer_features_simple(train_data, test_data, factor_list, label_col=None,
                                     epochs=10, batch_size=256):
    """
    简化版FTTransformer特征提取（使用默认参数）

    参数：
        train_data: 训练数据DataFrame
        test_data: 测试数据DataFrame
        factor_list: 特征列名列表
        label_col: 标签列名（可选）
        epochs: 训练轮数 (默认10)
        batch_size: 批大小 (默认256)

    返回：
        train_result: 添加特征后的训练数据
        test_result: 添加特征后的测试数据
        model: 训练好的模型

    默认参数适用于大多数场景，如需调优请使用add_fttransformer_features函数
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return add_fttransformer_features(
        train_data, test_data, factor_list, label_col,
        epochs=epochs, batch_size=batch_size, d_model=32,
        device=device
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 50)
    print("FTTRANSFORMER模块测试 (PyTorch原生实现)")
    print("=" * 50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")

    print("=" * 50)
    print("使用方法:")
    print("=" * 50)
    print("1. 简单调用（使用默认参数）:")
    print("   from dl_model_module import add_fttransformer_features_simple")
    print("   train_data, test_data, model = add_fttransformer_features_simple(")
    print("       train_data, test_data, factor_list, label")
    print("   )")
    print()
    print("2. 自定义参数:")
    print("   from dl_model_module import add_fttransformer_features")
    print("   train_data, test_data, model = add_fttransformer_features(")
    print("       train_data, test_data, factor_list, label,")
    print("       epochs=20, batch_size=256, d_model=64")
    print("   )")
    print("=" * 50)
    print("调参指南:")
    print("- d_model: 嵌入维度 (32/64/128)")
    print("- n_heads: 注意力头数 (4/8)")
    print("- num_layers: Transformer层数 (2/3/4)")
    print("- epochs: 训练轮数 (10-30)")
    print("- batch_size: 批大小 (128/256/512)")
    print("- lr: 学习率 (1e-4 到 1e-2)")
    print("=" * 50)
