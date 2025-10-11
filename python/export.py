import torch
import torch.nn as nn
import torch.onnx

class OptimizedDecisionModel(nn.Module):
    def __init__(self, num_groups=17, num_local=299, T_local=54, Ch=60, Te=250):
        super().__init__()
        self.num_groups = num_groups
        self.num_local = num_local
        self.T_local = T_local
        self.Ch = Ch
        self.Te = Te

        # 扩展参数维度以包含组信息 [group_dim, ...]
        # 全局参数 (原17个组的参数)
        self.W_global = nn.Parameter(torch.zeros(num_groups, Ch, 1))          # [17,60,1]
        self.Q_global = nn.Parameter(torch.zeros(num_groups, Te, 1))          # [17,250,1]
        self.b_global = nn.Parameter(torch.zeros(num_groups, 1))              # [17,1]
        self.Gamma_global = nn.Parameter(torch.zeros(num_groups))             # [17]
        self.Beta_global = nn.Parameter(torch.zeros(num_groups))              # [17]
        
        # 归一化参数
        self.register_buffer('M_global', torch.zeros(num_groups, Ch, 1, Te))     # [17,60,1,250]
        self.register_buffer('Sigma_global', torch.ones(num_groups, Ch, 1, Te))  # [17,60,1,250]
        self.register_buffer('M_local', torch.zeros(1, num_groups, num_local, T_local))  # [17,299,1,54]
        self.register_buffer('Sigma_local', torch.ones(1, num_groups, num_local, T_local))  # [17,299,1,54]
        
        # 局部模型参数
        self.gamma_local = nn.Parameter(torch.zeros(num_groups, num_local))   # [17,299]
        self.beta_local = nn.Parameter(torch.zeros(num_groups, num_local))    # [17,299]
        self.w_local = nn.Parameter(torch.zeros(num_groups, num_local, T_local, 1))  # [17,299,54,1]
        self.b_local = nn.Parameter(torch.zeros(num_groups, num_local, 1))   # [17,299,1]
        self.lr_model = nn.Parameter(torch.zeros(num_groups, num_local, 1))  # [17,299,1]
        self.gstf_weight = nn.Parameter(torch.ones(num_groups))              # [17]
        self.model_order = nn.Parameter(torch.ones(492))                   # [492]

    def forward(self, X_local, X_global):
        """
        X_local: [batch_size, T_local, num_local] = [1,54,299]
        X_global: [Ch, Te, batch_size] = [60,250,1]
        """
        
        # === 全局归一化 ===
        # X_global: [1,60,250,1] -> [17,250,1,60]  使得expand在第二维，避免CPU计算
        # M_global: [17,60,1,250] -> [17,250,1,60]
        X_global_preprocess = X_global.expand(-1,-1,-1,17).permute(3,2,0,1)
        X_global_BN = (X_global_preprocess - self.M_global.permute(0,3,2,1)) / \
                      torch.sqrt(self.Sigma_global.permute(0,3,2,1) + 1e-6)  # [17,250,1,60]
        
        # === 全局特征处理 ===
        # GSTF处理
        x = self.Gamma_global.view(-1,1,1,1) * X_global_BN + self.Beta_global.view(-1,1,1,1)  # [17,250,1,60]
        W_global_reshaped = self.W_global.unsqueeze(1)  # [17,1,60,1]
        h_global = torch.matmul(x, W_global_reshaped)   # [17,250,1,1]
        h_global = torch.matmul(
            h_global.permute(0,2,3,1),               # [17,1,1,250]
            self.Q_global.unsqueeze(1)               # [17,1,250,1]
        )  # [17,1,1,1]
        h_global = h_global + self.b_global.view(-1,1,1,1)
        h_global = self.gstf_weight.view(-1,1,1,1) * h_global  # [17,1,1,1]

        # === 局部归一化 ===
        # X_local: [1,1,54,299] -> [1,17,299,54]
        # M_local: [17,299,54] -> [17,299,1,54]
        P_trimmed = torch.eye(492)[:, self.model_order[:299].long()]   # [492,299]
        X_order = torch.matmul(X_local, P_trimmed)
        X_local_preprocess = X_order.expand(-1,17,-1,-1).permute(0,1,3,2)
        X_local_BN = (X_local_preprocess - self.M_local) / \
                    torch.sqrt(self.Sigma_local + 1e-6)  # [1,17,299,54]

        # === 局部特征处理 ===
        # 缩放与偏置 [17,299] -> [1,17,299,1]
        X_scaled = self.gamma_local.view(1,self.num_groups,-1,1) * X_local_BN + \
                 self.beta_local.view(1,self.num_groups,-1,1)  # [1,17,299,54]
        
        # 调整维度实现矩阵乘法
        f_batch = torch.matmul(
            X_scaled.permute(1,2,0,3),               # [17,299,1,54]
            self.w_local            # [17,299,54,1]
        )                           # [17,299,1,1]
        f_batch += self.b_local.view(self.num_groups,-1,1,1)  # 添加偏置 [17,299,1,1]
        f_batch = f_batch.permute(0,3,2,1)  # [17,1,1,299]
        
        # 加权求和
        weighted_sum = torch.matmul(
            f_batch,                    # [17,1,1,299]
            self.lr_model.unsqueeze(1)  # [17,1,299,1]
        )                               # [17,1,1,1]
        
        # === 结果融合 ===
        h_total = h_global + weighted_sum.permute(0,2,1,3)  # [17,1,1,1]
        s = torch.sigmoid(h_total)
        
        # 平均所有组的输出
        return s.mean(dim=0).squeeze(2)  # [1, 1]

class ParamInitializer:
    @staticmethod
    def load_parameters(model, param_path):
        params = torch.load(param_path, weights_only=True)
        # 参数重组逻辑（示例）
        # 假设原始参数按组存储
        model.W_global.data = params['W_global'].unsqueeze(-1)
        model.Q_global.data = params['Q_global'].unsqueeze(-1)
        model.b_global.data = params['b_global']
        model.Gamma_global.data = params['Gamma_global'].squeeze(-1)
        model.Beta_global.data = params['Beta_global'].squeeze(-1)

        model.gamma_local.data = params['gamma_local']
        model.beta_local.data = params['beta_local']
        model.w_local.data = params['w_local'].unsqueeze(-1)
        model.b_local.data = params['b_local'].unsqueeze(-1)
        model.lr_model.data = params['lr_model'].repeat(17, 1).unsqueeze(-1)
        model.gstf_weight.data = nn.Parameter(torch.ones(17))*0.3
        model.model_order.data = params['model_order']

        model.M_global.data = params['M_global'].unsqueeze(2)
        model.Sigma_global.data = params['Sigma_global'].unsqueeze(2)
        model.M_local.data = params['M_local'].unsqueeze(0)
        model.Sigma_local.data = params['Sigma_local'].unsqueeze(0)

def main():
    model = OptimizedDecisionModel()
    model.eval()
    
    # 参数加载（需要根据实际数据结构实现）
    ParamInitializer.load_parameters(model, 'converted_parameters.pt')
    
    # 验证推理
    dummy_local = torch.randn(1, 1, 54, 492)
    dummy_global = torch.randn(1, 60, 250, 1)
    with torch.no_grad():
        output = model(dummy_local, dummy_global)
        print("输出形状:", output.shape)  # [1,1]
    
    # 导出ONNX
    torch.onnx.export(
        model,
        (dummy_local, dummy_global),
        "optimized_model_v4_order.onnx",
        input_names=["X_local", "X_global"],
        output_names=["output"],
        opset_version=13
    )

if __name__ == "__main__":
    main()