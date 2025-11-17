# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ======================
# 📁 1. 加载模型和缩放参数
# ======================
save_dir = r'C:\Users\1\Desktop\GAN\saved_models'
G_path = os.path.join(save_dir, 'generator_lag.pth')
scaler_x_path = os.path.join(save_dir, 'scaler_x.npy')
scaler_y_path = os.path.join(save_dir, 'scaler_y.npy')

# 加载缩放器参数
scaler_x_params = np.load(scaler_x_path, allow_pickle=True).item()
scaler_y_params = np.load(scaler_y_path, allow_pickle=True).item()

scaler_x = StandardScaler()
scaler_x.mean_ = scaler_x_params['mean']
scaler_x.scale_ = scaler_x_params['scale']

scaler_y = StandardScaler()
scaler_y.mean_ = scaler_y_params['mean']
scaler_y.scale_ = scaler_y_params['scale']

# ======================
# 🧠 2. 定义生成器
# ======================
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
           'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']

noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                 'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 滞后特征（必须和训练时一致）
all_features = columns + noise_columns + [f'{col}_lag{lag}' for col in columns + noise_columns for lag in [1,2,3]]

class Generator(nn.Module):
    def __init__(self, input_dim=len(all_features), output_dim=len(columns)):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.model(x)

# 加载模型
G = Generator()
G.load_state_dict(torch.load(G_path))
G.eval()

print("✅ 模型与缩放器已加载完毕，可以进行预测。")

# ======================
# 📊 3. 加载测试数据
# ======================
test_data = pd.read_csv(r'C:\Users\1\Desktop\ML期末\数据集（含真实值）\modified_数据集Time_Series662.dat')

# --- 添加滞后特征 ---
def add_lag_features(df, cols, lags=[1,2,3]):
    df_lag = df.copy()
    for col in cols:
        for lag in lags:
            df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag).bfill()
    return df_lag

test_data = add_lag_features(test_data, columns + noise_columns)

X_test = scaler_x.transform(test_data[all_features].values)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ======================
# 🔮 4. 模型预测
# ======================
with torch.no_grad():
    Y_pred_scaled = G(X_test_tensor).numpy()
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

# ======================
# 💾 5. 保存预测结果（兼容评估脚本）
# ======================
# 将每一行的6个预测值转为字符串： '2.55 31.22 30.85 1.45 0.04 0.06'
pred_strs = [' '.join(map(str, row)) for row in Y_pred]
output_df = pd.DataFrame({'Predicted_Value': pred_strs})

output_path = r"C:\Users\1\Desktop\GAN\Output\GAN_lag_predictions.csv"
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 已保存预测结果至：{output_path}")
print(output_df.head())
