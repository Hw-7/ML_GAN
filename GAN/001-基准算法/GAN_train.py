# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import os

# ======================
#  1. 开始计时
# ======================
start_time = time.time()

# ======================
#  2. 数据加载
# ======================
train_data = pd.read_csv(r'C:\Users\1\Desktop\ML期末代码比赛\noised_data\modified_数据集Time_Series661_detail.dat')
test_data = pd.read_csv(r'C:\Users\1\Desktop\ML期末代码比赛\noised_data\modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
           'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']

noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                 'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# ======================
#  3. 添加滞后特征
# ======================
def add_lag_features(df, cols, lags=[1,2,3]):
    df_lag = df.copy()
    for col in cols:
        for lag in lags:
            df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag).bfill()
    return df_lag

train_data = add_lag_features(train_data, columns + noise_columns)
test_data = add_lag_features(test_data, columns + noise_columns)

all_features = columns + noise_columns + [f'{col}_lag{lag}' for col in columns + noise_columns for lag in [1,2,3]]

# ======================
#  4. 数据预处理
# ======================
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_x.fit_transform(train_data[all_features].values)
Y_train = scaler_y.fit_transform(train_data[columns].values)
X_test = scaler_x.transform(test_data[all_features].values)
Y_test = test_data[columns].values  # 原始值用于输出

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=128, shuffle=True)

# ======================
#  5. 定义 GAN 模型
# ======================
class Generator(nn.Module):
    def __init__(self, input_dim=X_train.shape[1], output_dim=len(columns)):
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

class Discriminator(nn.Module):
    def __init__(self, input_dim=X_train.shape[1] + len(columns)):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)

# ======================
# ️ 6. 初始化模型
# ======================
G = Generator()
D = Discriminator()

criterion_gan = nn.BCELoss()
criterion_l1 = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

lambda_l1 = 10
epochs = 80

# ======================
#  7. 模型保存路径
# ======================
save_dir = r'C:\Users\1\Desktop\GAN\saved_models'
os.makedirs(save_dir, exist_ok=True)
G_path = os.path.join(save_dir, 'generator_lag.pth')
D_path = os.path.join(save_dir, 'discriminator_lag.pth')
scaler_x_path = os.path.join(save_dir, 'scaler_x.npy')
scaler_y_path = os.path.join(save_dir, 'scaler_y.npy')

# ======================
#  8. 训练或加载模型
# ======================
if os.path.exists(G_path) and os.path.exists(D_path):
    G.load_state_dict(torch.load(G_path))
    D.load_state_dict(torch.load(D_path))
    print(" 已加载已有模型参数，无需重新训练。")
else:
    print(" 开始训练 GAN 模型 ...")
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            batch_size = x_batch.size(0)
            real = torch.ones((batch_size, 1))
            fake = torch.zeros((batch_size, 1))

            # --- 判别器 ---
            y_fake = G(x_batch).detach()
            D_real = D(x_batch, y_batch)
            D_fake = D(x_batch, y_fake)
            loss_D = 0.5 * (criterion_gan(D_real, real) + criterion_gan(D_fake, fake))

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # --- 生成器 ---
            y_fake = G(x_batch)
            D_fake = D(x_batch, y_fake)
            loss_G_gan = criterion_gan(D_fake, real)
            loss_G_l1 = criterion_l1(y_fake, y_batch)
            loss_G = loss_G_gan + lambda_l1 * loss_G_l1

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)
    np.save(scaler_x_path, {'mean': scaler_x.mean_, 'scale': scaler_x.scale_})
    np.save(scaler_y_path, {'mean': scaler_y.mean_, 'scale': scaler_y.scale_})
    print(f" 模型已保存至：{save_dir}")

# ======================
#  9. 预测阶段
# ======================
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_pred_scaled = G(X_test_tensor).detach().numpy()
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

# ======================
#  10. 保存结果
# ======================
results = []
for true_val, pred_val in zip(Y_test, Y_pred):
    error = np.abs(true_val - pred_val)
    results.append([
        ' '.join(map(str, true_val)),
        ' '.join(map(str, pred_val)),
        ' '.join(map(str, error))
    ])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv(r"C:\Users\1\Desktop\GAN\Output\GAN_lag_result.csv", index=False)
print("\n 已生成 GAN_lag_result.csv")

# ======================
#  11. 计算平均误差
# ======================
error_matrix = result_df["Error"].str.split(" ", expand=True).apply(pd.to_numeric)
mean_error_per_feature = error_matrix.mean()
final_avg_error = mean_error_per_feature.mean()

print("\n 每个特征的平均误差为：")
print(mean_error_per_feature)
print(f"\n 总体平均误差: {final_avg_error:.6f}")

# ======================
# ⏱ 12. 结束时间
# ======================
end_time = time.time()
print(f"\n⏱ 总耗时：{end_time - start_time:.2f} 秒")
