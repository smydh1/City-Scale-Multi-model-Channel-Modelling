import torch
import torch.nn as nn
import torch.nn.functional as F


#height输出feature和n_i
class Feature_and_n1(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Conv2d(in_channels, (feature_dim + 1), kernel_size=3, padding=1)
            nn.Conv2d(8, (feature_dim + 1), kernel_size=3, padding=1)
            #nn.Conv2d(Feature_and_Ptx, (feature_dim+1), kernel_size=3, padding=1)
        )
        self.feature_dim = feature_dim
        #self.n_head = nn.Conv2d(feature_dim, 1, kernel_size=1)  # 输出 n_i

    def forward(self, x):#
        feat_all= self.encoder(x)
        #n_i = self.n_head(feat)
        #feat = feat_all[:,0:-1]
        #n_i = feat_all[:,-1]
        
        #return feat, n_i
        #return feat_all[:,0:-1], feat_all[:,-1]
        return feat_all[:,0:int(self.feature_dim/2)], feat_all[:,int(self.feature_dim/2):self.feature_dim]
        
#satllite/osm输出feature和n_i
class Feature_and_n2(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, feature_dim+1, kernel_size=3, padding=1),
        )
        self.feature_dim = feature_dim
        #self.n_head = nn.Conv2d(feature_dim, 1, kernel_size=1) 
    def forward(self, x):
        feat_all = self.encoder(x)              # 输出特征
        #n_i = self.n_head(feat)             # 输出 n_i
        #return feat, n_i
        #return feat_all[:,0:-1], feat_all[:,-1]
        return feat_all[:,0:int(self.feature_dim/2)], feat_all[:,int(self.feature_dim/2):self.feature_dim]
        
    

#rssi输出feature和p_tx
class Feature_and_Ptx(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, feature_dim, kernel_size=3, padding=1),
        )
        self.feature_dim = feature_dim
        #self.ptx_head = nn.Conv2d(feature_dim, 1, kernel_size=1)

    def forward(self, x):
        assert x.dim() == 4
        feat_all = self.encoder(x)
        #p_tx_prime = self.ptx_head(feat)
        return feat_all[:,0:int(self.feature_dim/2)], feat_all[:,int(self.feature_dim/2):self.feature_dim]
        #return feat, p_tx_prime
    
#CI model     
class CILayerLocalAggregate(nn.Module):
    def __init__(self):
        super().__init__()
        #self.kernel_size = kernel_size
        #self.padding = kernel_size // 2

    def forward(self, p_tx, n, d):
        """
        p_tx, n, d: [B, 1, 256, 256]
        output: [B, 1, 256, 256]
        """
        d = torch.clamp(d, min=1.0)
        B, C, H, W = d.shape
        #K = self.kernel_size

        # unfold每个变量，得到滑动窗口内的patches
        # unfold后形状：[B, C*K*K, H*W]
        #p_tx_unf = F.unfold(p_tx, kernel_size=K, padding=self.padding)
        #n_unf    = F.unfold(n,    kernel_size=K, padding=self.padding)
        #d_unf    = F.unfold(d,    kernel_size=K, padding=self.padding)
        # 计算每个 patch 内的 prx
        # [B, K*K, H*W]
        # #prx_patch = p_tx_unf - 10 * n_unf * torch.log10(d_unf + 1e-6)
        # prx_patch = p_tx - 10 * n * torch.log10(d + 1e-6)
        # # 对每个 patch 求和 -> [B, H*W]
        # prx_sum = prx_patch.sum(dim=1, keepdim=True)
        # # reshape to [B, 1, H, W]
        # # then reshape to [B,H,W]
        # p_rx_hat = prx_sum.view(B, 1, H, W).squeeze()
        
        
        #prx_patch = p_tx_unf - 10 * n_unf * torch.log10(d_unf + 1e-6)
        prx_patch = p_tx - 10 * n * torch.log10(d + 1e-6)
        #prx_patch = 10 ** (prx_patch / 10)
        prx_patch=torch.pow(10, prx_patch / 10)
        # 对每个 patch 求和 -> [B, H*W]
        prx_sum = prx_patch.sum(dim=1, keepdim=True)
        prx_sum_db = 10 * torch.log10(prx_sum + 1e-12)
        p_rx_hat = prx_sum_db.view(B, 1, H, W).squeeze()

        
        return p_rx_hat


# 主模型：融合各输入，估计 d, n, p_tx，最终计算 p_rx_hat
class MultiModalCIModel_vM(nn.Module):
    def __init__(self, feature_dim=16, in_feature_dim = 16, sum_dim = 20):
        super().__init__()
        self.feature_dim = feature_dim
        self.in_feature_dim = in_feature_dim
        # RSSI
        self.nn_rssi = Feature_and_Ptx(in_channels=1, feature_dim=feature_dim)
        # Tx map
        #self.nn_tx_map = nn.Sequential(
        #    nn.Conv2d(in_feature_dim, 8, kernel_size=3, padding=1),
        #    nn.ReLU(),
         #   nn.Conv2d(8, feature_dim, kernel_size=3, padding=1),
        #)
        
        #self.nn_tx_map_part = nn.Conv2d(256*256, 8, kernel_size=3, padding=1)
        self.nn_tx_map_part = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.nn_tx_learned_feature_part = nn.Conv2d(int(in_feature_dim/2), 8, kernel_size=3, padding=1)
        self.nn_tx_fusion = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8+8, sum_dim, kernel_size=3, padding=1),
        )
        
        
        #  p_tx_prime + tx map 特征融合，估计 p_tx
        #self.head_p_tx = nn.Sequential(
        #    nn.Conv2d(feature_dim, 8, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(8, 1, kernel_size=1)  
        #)  
        
        # sat / osm / height：输出 feature + n_i
        self.nn_sat = Feature_and_n2(in_channels=3, feature_dim=feature_dim)
        self.nn_osm = Feature_and_n2(in_channels=3, feature_dim=feature_dim)
        self.nn_height = Feature_and_n1(in_channels=1, feature_dim=feature_dim)
        
        # 融合 n_i 得到最终 n（每像素）
        self.nn_n = nn.Sequential(
            nn.Conv2d(int(3*in_feature_dim/2), 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, sum_dim, kernel_size=1)
        )
        # 融合sat / osm / height所有特征，估计 d（每像素）
        self.nn_d = nn.Sequential(
            nn.Conv2d(int(2 * in_feature_dim), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, sum_dim, kernel_size=1)
        )
        #Ci layer init
        self.ci_layer = CILayerLocalAggregate()

        
        
        
    
    def forward(self, rssi, sat, osm, height, tx_map):
        
        # 1. RSSI: feature + p_tx_prime
        feat_rssi, p_tx_prime = self.nn_rssi(rssi)
        # 2. Tx map 
        #feat_tx = self.nn_tx_map(tx_map)
        tx_map_tmp = self.nn_tx_map_part(tx_map)
        tx_learned_feat_tmp = self.nn_tx_learned_feature_part(p_tx_prime)
        p_tx = self.nn_tx_fusion(torch.cat([tx_map_tmp, tx_learned_feat_tmp], dim=1) )
        # 3.  p_tx（p_tx_prime+tx map）
        #p_tx_input = torch.cat([p_tx_prime, feat_tx], dim=1)
        #p_tx = self.head_p_tx(p_tx_input)  
        # 4. sat/osm/height :feature + n_i
        feat_sat, n1 = self.nn_sat(sat)
        feat_osm, n2 = self.nn_osm(osm)
        feat_height, n3 = self.nn_height(height)
        # 5. n
        n_input = torch.cat([n1, n2, n3], dim=1)  
        n = self.nn_n(n_input)                   
        # 6.d
        feat_all = torch.cat([feat_rssi, feat_sat, feat_osm, feat_height], dim=1)
        d = self.nn_d(feat_all)  
        # 7.CI 
        p_rx_hat = self.ci_layer(p_tx, n, d) 
 

        return p_rx_hat


# # 主模型：融合各输入，估计 d, n, p_tx，最终计算 p_rx_hat
# class MultiModalCIModel(nn.Module):
#     def __init__(self, feature_dim=16):
#         super().__init__()
        
#         self.feature_dim = feature_dim
#         # RSSI
#         self.nn_rssi = Feature_and_Ptx(in_channels=1, feature_dim=feature_dim)
#         # Tx map
#         self.nn_tx_map = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(8, feature_dim, kernel_size=3, padding=1),
#         )
#         #  p_tx_prime + tx map 特征融合，估计 p_tx
#         self.head_p_tx = nn.Sequential(
#             nn.Conv2d(feature_dim + 1, 8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(8, 1, kernel_size=1)  
#         )  
        
#         # sat / osm / height：输出 feature + n_i
#         self.nn_sat = Feature_and_n2(in_channels=3, feature_dim=feature_dim)
#         self.nn_osm = Feature_and_n1(in_channels=1, feature_dim=feature_dim)
#         self.nn_height = Feature_and_n1(in_channels=1, feature_dim=feature_dim)
        
#         # 融合 n_i 得到最终 n（每像素）
#         self.nn_n = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(8, 1, kernel_size=1)
#         )
#         # 融合sat / osm / height所有特征，估计 d（每像素）
#         self.nn_d = nn.Sequential(
#             nn.Conv2d(4 * feature_dim, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 1, kernel_size=1)
#         )
#         #Ci layer init
#         self.ci_layer = CILayerLocalAggregate(kernel_size=30)

        
        
    
#     def forward(self, rssi, sat, osm, height, tx_map):
#         # 1. RSSI: feature + p_tx_prime
#         feat_rssi, p_tx_prime = self.nn_rssi(rssi)
#         # 2. Tx map 
#         feat_tx = self.nn_tx_map(tx_map)
#         # 3.  p_tx（p_tx_prime+tx map）
#         p_tx_input = torch.cat([p_tx_prime, feat_tx], dim=1)
#         p_tx = self.head_p_tx(p_tx_input)  
#         # 4. sat/osm/height :feature + n_i
#         feat_sat, n1 = self.nn_sat(sat)
#         feat_osm, n2 = self.nn_osm(osm)
#         feat_height, n3 = self.nn_height(height)
#         # 5. n
#         n_input = torch.cat([n1, n2, n3], dim=1)  
#         n = self.nn_n(n_input)                   
#         # 6.d
#         feat_all = torch.cat([feat_rssi, feat_sat, feat_osm, feat_height], dim=1)
#         d = self.nn_d(feat_all)  
#         # 7.CI 
#         p_rx_hat = self.ci_layer(p_tx, n, d) 
 

#         return p_rx_hat
         




        
        
       