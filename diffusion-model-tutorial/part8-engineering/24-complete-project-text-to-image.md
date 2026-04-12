# 第二十四章：完整项目——文本到图像生成系统

> **本章导读**：综合前23章所学，本章构建一个完整的文本到图像生成系统。从零实现一个微型扩散模型（64×64像素，可在消费级GPU/CPU上训练），然后展示如何使用🤗 Diffusers库运行Stable Diffusion完整推理管线，并介绍LoRA微调、ControlNet控制等实用技术。

**前置知识**：全部前章内容
**预计学习时间**：150分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 从零实现一个完整的迷你文本条件扩散模型（训练+采样）
2. 使用🤗 Diffusers库运行SD完整推理管线
3. 实施LoRA微调定制化扩散模型
4. 理解ControlNet的架构原理并使用预训练ControlNet
5. 设计完整的文本到图像系统（从需求到部署）

---

## 24.1 完整系统架构回顾

### 组件清单

一个完整的文本到图像系统包含以下组件：

```
文本输入 (prompt)
    ↓
[文本编码器] (CLIP/T5)
    ↓ 文本嵌入 (B, L, d)
[U-Net / DiT] ←── 时间步嵌入
    ↑ 潜在噪声 x_T
[VAE编码器] ← 可选（图像条件）
    
[噪声调度器] + [采样算法] (DDIM/DPM-Solver)
    ↓ 潜在样本 x_0
[VAE解码器]
    ↓
生成图像 (512×512 或 1024×1024)
```

### 两种实现路径

1. **从零实现**（本章前半）：在迷你数据集上训练小型扩散模型
2. **使用Diffusers**（本章后半）：运行、微调、扩展预训练SD模型

---

## 24.2 迷你扩散模型：从零实现

### 目标

训练一个64×64灰度图像的条件扩散模型，在FashionMNIST数据集上生成特定类别的衣物图像。

- **数据集**：FashionMNIST（28×28灰度图像，10类别，上采样到32×32）
- **模型**：简化U-Net（约5M参数）
- **训练**：单GPU，约30分钟收敛
- **条件**：类别标签（通过Embedding注入）

---

## 完整代码实战

```python
"""
第二十四章：完整迷你扩散模型
在FashionMNIST上训练类别条件扩散模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
from typing import Optional, Tuple


# ============================================================
# 1. 噪声调度器
# ============================================================

class DDPMScheduler:
    """DDPM噪声调度器（余弦调度）"""
    
    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        # 余弦调度
        t = torch.arange(T + 1, dtype=torch.float32) / T
        f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bars = (f / f[0]).clamp(min=1e-8)
        
        self.alpha_bars = alpha_bars[:T]                          # shape: (T,)
        alpha_bars_prev = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])
        self.alphas = self.alpha_bars / alpha_bars_prev           # shape: (T,)
        self.betas = (1.0 - self.alphas).clamp(max=0.999)        # shape: (T,)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        正向加噪
        
        Args:
            x0: 原始图像，shape (B, C, H, W)
            t: 时间步，shape (B,)
        
        Returns:
            (x_t, noise): 加噪图像和噪声，各 shape (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_ab = self.alpha_bars[t].sqrt()[:, None, None, None]      # (B, 1, 1, 1)
        sqrt_1mab = (1 - self.alpha_bars[t]).sqrt()[:, None, None, None]
        
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise
    
    def ddim_step(self, x_t: torch.Tensor, t: int, t_prev: int,
                  eps_pred: torch.Tensor) -> torch.Tensor:
        """
        DDIM确定性采样步
        
        Args:
            x_t: 当前状态，shape (B, C, H, W)
            t, t_prev: 当前和前一时间步
            eps_pred: 预测噪声，shape (B, C, H, W)
        
        Returns:
            x_prev: shape (B, C, H, W)
        """
        ab_t = self.alpha_bars[t]
        ab_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # 预测 x0
        x0_pred = (x_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        
        # DDIM更新
        return ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred


# ============================================================
# 2. 迷你U-Net（含类别条件）
# ============================================================

class SinusoidalEmbed(nn.Module):
    """正弦位置编码（用于时间步）"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步，shape (B,)
        Returns:
            emb: shape (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                                                  # (half,)
        args = t[:, None].float() * freqs[None, :]                        # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)                 # (B, dim)
        return emb


class ResBlock(nn.Module):
    """残差块（含时间步条件）"""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, in_ch, H, W)
            t_emb: shape (B, time_dim)
        Returns:
            out: shape (B, out_ch, H, W)
        """
        h = self.conv1(x)                                             # (B, out_ch, H, W)
        h = h + self.time_proj(t_emb)[:, :, None, None]              # 加时间条件
        h = self.conv2(h)
        return h + self.skip(x)


class MiniUNet(nn.Module):
    """
    迷你U-Net（用于32×32图像，类别条件）
    参数量约5M
    """
    
    def __init__(self, in_ch: int = 1, n_classes: int = 10,
                 base_ch: int = 64, T: int = 1000):
        super().__init__()
        time_dim = base_ch * 4
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalEmbed(base_ch),
            nn.Linear(base_ch, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 类别嵌入（CFG：+1个null类别）
        self.class_embed = nn.Embedding(n_classes + 1, time_dim)
        
        # 编码器
        self.enc1 = ResBlock(in_ch, base_ch, time_dim)              # 32×32
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)        # 16×16
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)
        
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim)    # 8×8
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1)
        
        # 中间块
        self.mid = ResBlock(base_ch * 4, base_ch * 4, time_dim)     # 4×4
        
        # 解码器（含跳跃连接）
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ResBlock(base_ch * 8, base_ch * 2, time_dim)    # 8×8（含跳跃）
        
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ResBlock(base_ch * 4, base_ch, time_dim)        # 16×16
        
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)        # 32×32
        
        # 输出头
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 噪声图像，shape (B, 1, 32, 32)
            t: 时间步，shape (B,)
            y: 类别标签，shape (B,)；None时用null类别（CFG无条件）
        
        Returns:
            eps_pred: 预测噪声，shape (B, 1, 32, 32)
        """
        B = x.shape[0]
        
        # 时间嵌入 + 类别嵌入
        t_emb = self.time_embed(t)                                   # (B, time_dim)
        
        null_class = torch.full((B,), 10, dtype=torch.long, device=x.device)  # null=10
        y_idx = null_class if y is None else y
        c_emb = self.class_embed(y_idx)                              # (B, time_dim)
        
        cond = t_emb + c_emb                                         # 简单相加
        
        # 编码器
        e1 = self.enc1(x, cond)                                      # (B, 64, 32, 32)
        e2 = self.enc2(self.down1(e1), cond)                         # (B, 128, 16, 16)
        e3 = self.enc3(self.down2(e2), cond)                         # (B, 256, 8, 8)
        
        # 中间
        m = self.mid(self.down3(e3), cond)                           # (B, 256, 4, 4)
        
        # 解码器
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1), cond)   # (B, 128, 8, 8)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), cond)  # (B, 64, 16, 16)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)  # (B, 64, 32, 32)
        
        return self.out(d1)                                          # (B, 1, 32, 32)


# ============================================================
# 3. 训练循环
# ============================================================

def train_fashion_diffusion(n_epochs: int = 50, device: str = 'cpu',
                             p_uncond: float = 0.1) -> MiniUNet:
    """
    在FashionMNIST上训练类别条件扩散模型
    
    Args:
        n_epochs: 训练轮数
        device: 'cpu' 或 'cuda'
        p_uncond: CFG条件Dropout概率
    
    Returns:
        训练好的模型
    """
    T = 1000
    scheduler = DDPMScheduler(T=T)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # 归一化到[-1, 1]
    ])
    
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    
    model = MiniUNet(in_ch=1, n_classes=10, base_ch=64, T=T).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # EMA（指数移动平均）
    ema_model = MiniUNet(in_ch=1, n_classes=10, base_ch=64, T=T).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.995
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for x, y in loader:
            x = x.to(device)                                        # (B, 1, 32, 32)
            y = y.to(device)                                        # (B,)
            B = x.shape[0]
            
            # 随机时间步
            t = torch.randint(0, T, (B,), device=device)
            
            # 加噪
            x_t, noise = scheduler.q_sample(x, t)
            
            # CFG：条件Dropout
            y_input = y.clone()
            mask = torch.rand(B, device=device) < p_uncond
            y_input[mask] = 10  # null类别
            
            # 预测
            eps_pred = model(x_t, t, y_input)                       # (B, 1, 32, 32)
            
            # 损失（可选：Min-SNR加权）
            loss = F.mse_loss(eps_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新EMA
            for p, p_ema in zip(model.parameters(), ema_model.parameters()):
                p_ema.data.lerp_(p.data, 1 - ema_decay)
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler_lr.step()
        avg_loss = total_loss / n_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, "
                  f"lr={scheduler_lr.get_last_lr()[0]:.6f}")
    
    return ema_model  # 返回EMA模型（通常比瞬时模型更好）


# ============================================================
# 4. DDIM采样（含CFG）
# ============================================================

@torch.no_grad()
def sample_ddim_cfg(model: MiniUNet, scheduler: DDPMScheduler,
                    class_labels: torch.Tensor, guidance_scale: float = 7.5,
                    num_steps: int = 50, device: str = 'cpu') -> torch.Tensor:
    """
    DDIM + CFG 采样
    
    Args:
        class_labels: 类别标签，shape (B,)
        guidance_scale: CFG引导强度
        num_steps: 采样步数
    
    Returns:
        images: 生成图像，shape (B, 1, 32, 32)，值域[-1, 1]
    """
    B = class_labels.shape[0]
    x = torch.randn(B, 1, 32, 32, device=device)                   # 从噪声开始
    
    T = scheduler.T
    step_size = T // num_steps
    timesteps = list(range(T - 1, 0, -step_size))
    
    for i, t_val in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
        
        # CFG：条件和无条件预测
        eps_cond = model(x, t_batch, class_labels)                  # (B, 1, 32, 32)
        eps_uncond = model(x, t_batch, None)                        # (B, 1, 32, 32)
        
        # CFG混合
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        # DDIM步
        x = scheduler.ddim_step(x, t_val, t_prev, eps)
    
    return x


# ============================================================
# 5. 使用Diffusers（简化演示）
# ============================================================

DIFFUSERS_DEMO = '''
# 使用🤗 Diffusers运行Stable Diffusion
# pip install diffusers transformers accelerate

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,          # BF16加速
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config               # 替换调度器为DPM-Solver++
)
pipe = pipe.to("cuda")

# 可选优化
pipe.enable_xformers_memory_efficient_attention()  # Flash Attention
# pipe.enable_model_cpu_offload()       # CPU Offload（节省显存）

# 生成图像
images = pipe(
    prompt="a beautiful sunset over mountains, highly detailed, 8k",
    negative_prompt="blurry, low quality, watermark",
    num_inference_steps=20,             # DPM-Solver++ 20步
    guidance_scale=7.5,                 # CFG强度
    num_images_per_prompt=4,
).images

# 保存
for i, img in enumerate(images):
    img.save(f"output_{i}.png")
'''

LORA_DEMO = '''
# LoRA微调示例
# pip install diffusers peft

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch

# 加载基础模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

# LoRA配置
lora_config = LoraConfig(
    r=16,                               # LoRA rank（越大容量越高）
    lora_alpha=16,
    target_modules=["to_q", "to_v"],    # 对Q和V投影层添加LoRA
    lora_dropout=0.0,
    bias="none",
)

# 对U-Net添加LoRA
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.print_trainable_parameters()

# 训练（只训练LoRA参数，约1-2%参数量）
# ... 训练循环同普通扩散模型训练 ...

# 保存LoRA权重
pipe.unet.save_pretrained("my_lora_weights/")

# 加载并使用
# pipe.unet.load_adapter("my_lora_weights/")
'''

CONTROLNET_DEMO = '''
# ControlNet使用示例
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch

# 加载ControlNet（Canny边缘条件）
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 准备控制图像（Canny边缘）
import cv2
import numpy as np
img = np.array(Image.open("input.jpg"))
edges = cv2.Canny(img, 100, 200)
control_image = Image.fromarray(edges)

# 生成（受边缘控制的图像）
images = pipe(
    prompt="a beautiful painting, highly detailed",
    image=control_image,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,  # ControlNet强度
).images

images[0].save("output_controlnet.png")
'''


# ============================================================
# 6. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("迷你扩散模型：FashionMNIST类别条件生成")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 验证模型结构
    model = MiniUNet(in_ch=1, n_classes=10, base_ch=64, T=1000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    
    # 测试前向传播
    x = torch.randn(4, 1, 32, 32)
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 10, (4,))
    eps = model(x, t, y)
    print(f"前向传播: 输入{x.shape} → 输出{eps.shape}")
    
    # 测试采样（未训练）
    scheduler = DDPMScheduler(T=1000)
    labels = torch.tensor([0, 1, 2, 3])  # T-shirt, Trouser, Pullover, Dress
    samples = sample_ddim_cfg(model, scheduler, labels, guidance_scale=7.5, num_steps=20)
    print(f"采样输出: {samples.shape}")
    print(f"样本范围: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # 可选：完整训练（注释掉以节省时间）
    # print("\n开始训练（约30分钟，可跳过）...")
    # trained_model = train_fashion_diffusion(n_epochs=50, device=device)
    
    print("\n" + "=" * 60)
    print("Diffusers使用示例（代码片段）")
    print("=" * 60)
    print("1. 基础SD推理:")
    print(DIFFUSERS_DEMO)
    print("\n2. LoRA微调:")
    print(LORA_DEMO)
    print("\n3. ControlNet:")
    print(CONTROLNET_DEMO)
    
    print("\n" + "=" * 60)
    print("系统设计清单")
    print("=" * 60)
    print("""
完整文本到图像系统需要考虑：

[模型选择]
□ 基础模型: SD1.5 / SDXL / SD3 / FLUX（按显存/质量要求）
□ 是否需要微调: LoRA / DreamBooth / 全量微调
□ 条件控制: ControlNet / IP-Adapter / T2I-Adapter

[推理优化]
□ 精度: BF16 / FP16（节省50%显存）
□ 注意力: Flash Attention / xFormers
□ 编译: torch.compile（+1.5x速度）
□ 步数: DDIM 50步 → DPM-Solver++ 20步 → LCM 4步
□ 显存: CPU Offload / 模型切片 / 动态分辨率

[部署架构]
□ API服务: FastAPI + 异步推理队列
□ 批处理: 动态批次合并
□ 缓存: 文本嵌入缓存（相同prompt复用）
□ 监控: 推理延迟 / OOM报警 / 内容安全过滤

[内容安全]
□ NSFW过滤（Safety Checker）
□ 提示词过滤
□ 水印嵌入
    """)


if __name__ == "__main__":
    main()
```

---

## 24.3 LoRA微调原理

### 低秩分解

LoRA（Low-Rank Adaptation，Hu et al. 2021）将权重更新分解为低秩矩阵乘积：

$$W = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**参数量**：原始 $W \in \mathbb{R}^{d \times k}$ 有 $dk$ 个参数；LoRA只有 $r(d+k)$ 个参数。

典型值 $r = 16$，$d = k = 768$（CLIP维度）：
- 原始：$768 \times 768 = 590K$
- LoRA：$16 \times 1536 = 24K$（约4%）

### 在扩散模型中的应用

通常对U-Net的注意力层（$W_Q, W_V$）或全连接层添加LoRA。常见用途：
- **风格微调**：用100张特定风格图片微调（DreamBooth+LoRA）
- **人物定制**：学习特定人物外貌
- **任务特化**：提高特定域的生成质量

---

## 24.4 ControlNet

### 架构原理

Zhang et al. (2023) 的ControlNet在SD U-Net基础上添加可训练的副本（"ControlNet分支"），以结构化条件（边缘、深度、姿态等）控制生成：

1. 复制U-Net的编码器部分
2. 添加"Zero Convolution"层（初始化为零，训练开始时不影响主网络）
3. 控制条件通过副本处理后加到主U-Net

**Zero Convolution的意义**：训练初始时副本输出为零（完全等于原始SD），随着训练深入，副本学习如何注入控制信号，避免破坏预训练权重。

---

## 本章小结

| 组件 | 本教程实现 | 工业实现 |
|------|----------|----------|
| 数据 | FashionMNIST (32×32) | LAION-5B (512×512) |
| 文本条件 | 类别Embedding | CLIP/T5文本编码器 |
| 模型 | 迷你U-Net (5M) | SD U-Net (860M) / DiT |
| 潜在空间 | 像素空间 | VAE潜在空间 (×8压缩) |
| 采样 | DDIM + CFG (50步) | DPM-Solver++ (20步) |
| 训练 | 单GPU, 30分钟 | 256 GPU × 数天 |

---

## 练习题

### 基础题

**24.1** 修改迷你U-Net，将类别条件替换为文本条件（使用一个简单的文本编码器，如字符级Embedding + 平均池化），在FashionMNIST上训练，评估文本描述（"白色T恤"）能否生成对应图像。

**24.2** 实现EMA（指数移动平均）：在训练循环中维护EMA模型参数，对比EMA模型和瞬时模型在50步和10步采样时的FID分数差异。

### 中级题

**24.3** 在迷你扩散模型中实现LoRA：仅训练U-Net注意力层的LoRA权重（保持其余层冻结），在FashionMNIST的一个子集（如仅T-shirt类）上微调，评估LoRA是否能让模型专注生成T-shirt。

**24.4** 使用🤗 Diffusers构建一个简单的批量文本到图像服务：给定一个提示词列表，并发生成图像，实现请求排队和批次合并（Dynamic Batching），测量不同并发度下的吞吐量。

### 提高题

**24.5** 完整项目：从零实现一个32×32的ControlNet（以sobel边缘图为条件），在FashionMNIST上训练，验证：给定T-shirt的边缘图，能否生成结构相似但样式不同的衣物图像。关键：实现Zero Convolution初始化和编码器复制。

---

## 练习答案

**24.1** 关键修改：将`class_embed = Embedding(n_classes+1, ...)`替换为字符级`token_embed`，平均池化后作为条件。字符级模型通常需要更多训练步数，且条件对齐比类别ID弱（因为文本理解能力有限）。

**24.2** EMA关键代码：`for p, p_ema in zip(model.parameters(), ema_model.parameters()): p_ema.data = decay * p_ema.data + (1-decay) * p.data`。EMA通常比瞬时模型FID低1-2点（生成更稳定）。

**24.3** LoRA关键代码：在`nn.Linear`旁添加`A = nn.Linear(in, r, bias=False); B = nn.Linear(r, out, bias=False)`，初始化`B=0`。微调后，仅T-shirt的IS（Inception Score）应明显提高。

**24.4** Dynamic Batching核心：异步queue + 定时flush（每100ms或batch满时触发推理），将多个请求合并为一个批次。

**24.5** Zero Convolution：`nn.Conv2d(...); torch.nn.init.zeros_(conv.weight); torch.nn.init.zeros_(conv.bias)`。训练时，在边缘图条件下，生成图像应保留主要轮廓结构。

---

## 延伸阅读

1. **Ho et al. (2020)**. *Denoising Diffusion Probabilistic Models* — DDPM原文
2. **Rombach et al. (2022)**. *High-Resolution Image Synthesis with Latent Diffusion Models* — Stable Diffusion
3. **Hu et al. (2021)**. *LoRA: Low-Rank Adaptation of Large Language Models* — LoRA（扩散模型同样适用）
4. **Zhang et al. (2023)**. *Adding Conditional Control to Text-to-Image Diffusion Models* — ControlNet
5. **🤗 Diffusers文档**. https://huggingface.co/docs/diffusers — 实用API参考

---

[← 上一章：推理优化工程实践](./23-inference-optimization.md)

[附录A：数学速查表 →](../appendix/math-reference.md)

[返回目录](../README.md)
