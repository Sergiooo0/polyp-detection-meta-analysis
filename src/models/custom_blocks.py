import torch
import torch.nn as nn
import torch.nn.functional as F

class TST(nn.Module):
    def __init__(self, c_local, c_global, num_heads=4):
        super().__init__()
        
        # Dimensiones especificadas en el paper YOLOv5_TST (Sec 3.3)
        # Dicen: Q/K dim = 4, V dim = 16. Esto es MUY pequeño (para ser ligero).
        self.head_dim_qk = 4
        self.head_dim_v = 16
        self.num_heads = num_heads
        # Proyecciones para Query (del local), Key y Value (del global)
        # Nota: Usamos c_local para Query y c_global para K/V
        self.project_q = nn.Conv2d(c_local, num_heads * self.head_dim_qk, 1)
        self.project_k = nn.Conv2d(c_global, num_heads * self.head_dim_qk, 1)
        self.project_v = nn.Conv2d(c_global, num_heads * self.head_dim_v, 1)
        
        # Salida de la atención (proyección final para volver a dimensión local)
        self.project_out = nn.Conv2d(num_heads * self.head_dim_v, c_local, 1)
        
        # Feed Forward Network (FFN) mencionado en Sec 3.2
        # Típicamente expande y contrae. El paper menciona ReLU y BatchNorm.
        self.ffn = nn.Sequential(
            nn.Conv2d(c_local, c_local * 2, 1), # Expansión arbitraria (usualmente x2 o x4)
            nn.BatchNorm2d(c_local * 2),
            nn.ReLU(),
            nn.Conv2d(c_local * 2, c_local, 1),
            nn.BatchNorm2d(c_local)
        )

    def forward(self, x_local, x_global):
        """
        x_local: Feature map de alta resolución [Batch, C_l, H, W]
        x_global: Feature map de baja resolución (contexto global) [Batch, C_g, H_g, W_g]
        """
        B, Cl, H, W = x_local.shape
        _, Cg, Hg, Wg = x_global.shape

        # 1. Preparar Q, K, V
        # Query viene del Local
        q = self.project_q(x_local) # [B, heads*4, H, W]
        
        # Key y Value vienen del Global
        k = self.project_k(x_global) # [B, heads*4, Hg, Wg]
        v = self.project_v(x_global) # [B, heads*16, Hg, Wg]

        # Aplanar espacialmente para atención
        q = q.view(B, self.num_heads, self.head_dim_qk, H * W).permute(0, 1, 3, 2) # [B, Heads, HW, 4]
        k = k.view(B, self.num_heads, self.head_dim_qk, Hg * Wg) # [B, Heads, 4, HgWg]
        v = v.view(B, self.num_heads, self.head_dim_v, Hg * Wg).permute(0, 1, 3, 2) # [B, Heads, HgWg, 16]

        # 2. Atención (Scaled Dot-Product)
        attn_score = torch.matmul(q, k) # [B, Heads, HW, HgWg]
        attn_score = attn_score / (self.head_dim_qk ** 0.5)
        attn_probs = F.softmax(attn_score, dim=-1)

        # 3. Aggregation
        attn_out = torch.matmul(attn_probs, v) # [B, Heads, HW, 16]
        
        # Reshape para volver a formato imagen
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, self.num_heads * self.head_dim_v, H, W)
        
        # 4. Proyección de salida + Residual Connection (Local puro + Info Global)
        out = self.project_out(attn_out)
        out = x_local + out # Residual connection (Sumamos la info global a la local)
        
        # 5. Feed Forward
        out = out + self.ffn(out) # Otra residual connection típica en Transformers
        
        return out