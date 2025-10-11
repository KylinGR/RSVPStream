import torch
import numpy as np

# 加载训练好的模型文件
checkpoint = torch.load('Model_1_MG_50.pt', weights_only=False, map_location=torch.device('cpu'))

# ----------------- 参数提取阶段 -----------------
# 0. 提取全局归一化参数
M_global = checkpoint['M_global'].numpy()          # [60,250,17]
Sigma_global = checkpoint['Sigma_global'].numpy()  # [60,250,17]
M_local = checkpoint['M_local'].numpy()            # [299,54,17]
Sigma_local = checkpoint['Sigma_local'].numpy()    # [299,54,17]

# 1. 提取local_weight参数 (492,1) -> 取前299
lr_model = checkpoint['local_weight'][:299].squeeze(1).numpy()  # [299]
model_order = checkpoint['Model_order'].numpy()

# 2. 提取17个组的参数
model_list = checkpoint['Model']
all_groups = []

converted_params = {}

for group_idx in range(17):  # 显式遍历17个组
    group = model_list[group_idx]
    gstf_params = group[0]
    local_models = group[1:300]  # 取299个local模型
    
    # 2.1 提取全局参数
    W_global = gstf_params.W_global.squeeze().detach().numpy()   # [60]
    Q_global = gstf_params.Q_global.squeeze().detach().numpy()   # [250]
    b_global = gstf_params.b_global.detach().numpy()              # scalar
    Gamma_global = gstf_params.Gamma_global.detach().numpy()      # scalar
    Beta_global = gstf_params.Beta_global.detach().numpy()        # scalar
    
    # 2.2 提取当前组的归一化参数
    group_M_global = M_global[:, :, group_idx]          # [60,250]
    group_Sigma_global = Sigma_global[:, :, group_idx]  # [60,250]
    group_M_local = M_local[:, :, group_idx]            # [299,54]
    group_Sigma_local = Sigma_local[:, :, group_idx]    # [299,54]
    
    # 2.3 合并局部模型参数
    w_local = np.zeros((299, 54))    # 合并后形状
    b_local = np.zeros(299)
    gamma_local = np.zeros(299)
    beta_local = np.zeros(299)
    
    for local_idx, local_model in enumerate(local_models):
        w_local[local_idx] = local_model.w_local.squeeze().detach().numpy()  # [54]
        b_local[local_idx] = local_model.b_local.detach().numpy()
        gamma_local[local_idx] = local_model.gamma_local.detach().numpy()
        beta_local[local_idx] = local_model.beta_local.detach().numpy()
    
    if 'W_global' in converted_params:
        converted_params['W_global'] = torch.cat([converted_params['W_global'], torch.from_numpy(W_global).float().unsqueeze(0)], dim=0)
    else:
        converted_params['W_global'] = torch.from_numpy(W_global).float().unsqueeze(0)
    
    if 'Q_global' in converted_params:
        converted_params['Q_global'] = torch.cat([converted_params['Q_global'], torch.from_numpy(Q_global).float().unsqueeze(0)], dim=0)
    else:
        converted_params['Q_global'] = torch.from_numpy(Q_global).float().unsqueeze(0)
    
    if 'b_global' in converted_params:
        converted_params['b_global'] = torch.cat([converted_params['b_global'], torch.from_numpy(b_global).float().unsqueeze(0)], dim=0)
    else:
        converted_params['b_global'] = torch.from_numpy(b_global).float().unsqueeze(0)
    
    if 'Gamma_global' in converted_params:
        converted_params['Gamma_global'] = torch.cat([converted_params['Gamma_global'], torch.from_numpy(Gamma_global).float().unsqueeze(0)], dim=0)
    else:
        converted_params['Gamma_global'] = torch.from_numpy(Gamma_global).float().unsqueeze(0)
    
    if 'Beta_global' in converted_params:
        converted_params['Beta_global'] = torch.cat([converted_params['Beta_global'], torch.from_numpy(Beta_global).float().unsqueeze(0)], dim=0)
    else:
        converted_params['Beta_global'] = torch.from_numpy(Beta_global).float().unsqueeze(0)
    
    if 'w_local' in converted_params:
        converted_params['w_local'] = torch.cat([converted_params['w_local'], torch.from_numpy(w_local).float().unsqueeze(0)], dim=0)
    else:
        converted_params['w_local'] = torch.from_numpy(w_local).float().unsqueeze(0)

    if 'b_local' in converted_params:
        converted_params['b_local'] = torch.cat([converted_params['b_local'], torch.from_numpy(b_local).float().unsqueeze(0)], dim=0)
    else:
        converted_params['b_local'] = torch.from_numpy(b_local).float().unsqueeze(0)

    if 'gamma_local' in converted_params:
        converted_params['gamma_local'] = torch.cat([converted_params['gamma_local'], torch.from_numpy(gamma_local).float().unsqueeze(0)], dim=0)
    else:
        converted_params['gamma_local'] = torch.from_numpy(gamma_local).float().unsqueeze(0)

    if 'beta_local' in converted_params:
        converted_params['beta_local'] = torch.cat([converted_params['beta_local'], torch.from_numpy(beta_local).float().unsqueeze(0)], dim=0)
    else:
        converted_params['beta_local'] = torch.from_numpy(beta_local).float().unsqueeze(0)
    
converted_params['lr_model'] = torch.from_numpy(lr_model).float()
converted_params['model_order'] = torch.from_numpy(model_order).float()
converted_params['M_global'] = torch.from_numpy(M_global).float().permute(2, 0, 1)
converted_params['Sigma_global'] = torch.from_numpy(Sigma_global).float().permute(2, 0, 1)
converted_params['M_local'] = torch.from_numpy(M_local).float().permute(2, 0, 1)
converted_params['Sigma_local'] = torch.from_numpy(Sigma_local).float().permute(2, 0, 1)

# 保存为.pt文件
torch.save(converted_params, 'converted_parameters.pt')

print("参数已成功保存到converted_parameters.pt")