# 附录B：🤗 Diffusers API 速查

> 本附录汇总🤗 Diffusers库的常用API，覆盖模型加载、推理管线、调度器、微调工具等，方便工程实践时快速查阅。

---

## B.1 安装与环境

```bash
pip install diffusers transformers accelerate xformers safetensors
pip install peft  # LoRA支持

# GPU版本（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**版本兼容**（截至2024年）：
- diffusers >= 0.27
- transformers >= 4.38
- torch >= 2.1

---

## B.2 基础推理管线

### StableDiffusionPipeline

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载（FP16）
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,           # 关闭安全检查器
    requires_safety_checker=False,
)
pipe = pipe.to("cuda")

# 基础推理
result = pipe(
    prompt="a golden retriever sitting in a field of flowers",
    negative_prompt="blurry, distorted, low quality",
    height=512, width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    num_images_per_prompt=2,
    generator=torch.Generator("cuda").manual_seed(42),  # 固定随机种子
)
images = result.images  # list of PIL.Image
images[0].save("output.png")
```

### SDXL推理

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

result = pipe(
    prompt="a majestic lion, photorealistic, 8k",
    height=1024, width=1024,
    num_inference_steps=30,
    guidance_scale=5.0,
)
```

---

## B.3 调度器（Sampler）

### 调度器列表

| 类名 | 算法 | 推荐步数 | 特点 |
|------|------|----------|------|
| `DDPMScheduler` | DDPM | 1000 | 原始，慢 |
| `DDIMScheduler` | DDIM | 50 | 确定性 |
| `DPMSolverMultistepScheduler` | DPM-Solver++ | 20 | 高精度 |
| `UniPCMultistepScheduler` | UniPC | 10-20 | 快速 |
| `EulerDiscreteScheduler` | Euler | 30 | 通用 |
| `EulerAncestralDiscreteScheduler` | Euler-a | 30 | 随机性更高 |
| `LCMScheduler` | LCM | 4 | 极速 |

### 切换调度器

```python
from diffusers import DPMSolverMultistepScheduler

# 使用当前模型配置初始化DPM-Solver++
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",  # 或 "sde-dpmsolver++"（随机版）
    use_karras_sigmas=True,        # Karras噪声调度（提升质量）
)
```

---

## B.4 内存优化API

```python
# 启用Flash Attention（需安装xformers）
pipe.enable_xformers_memory_efficient_attention()

# CPU Offload（节省GPU显存，速度略慢）
pipe.enable_model_cpu_offload()           # 逐模块offload
pipe.enable_sequential_cpu_offload()      # 逐层offload（最省显存）

# VAE优化（高分辨率图像）
pipe.enable_vae_slicing()                 # 批维度切片
pipe.enable_vae_tiling()                  # 空间维度切片

# torch.compile（PyTorch 2.0+）
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# 混合精度
pipe.unet.to(torch.bfloat16)
```

---

## B.5 图像到图像（img2img）

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

init_image = Image.open("input.jpg").resize((512, 512))

result = pipe(
    prompt="a dog sitting in a painting, oil on canvas",
    image=init_image,
    strength=0.75,          # 0=保持原图，1=完全重新生成
    guidance_scale=7.5,
    num_inference_steps=30,
)
result.images[0].save("img2img_output.png")
```

---

## B.6 图像修复（Inpainting）

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

image = Image.open("image.jpg").resize((512, 512))
# 白色区域=修复区域，黑色=保留区域
mask = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))  # 全黑（示例）

result = pipe(
    prompt="a beautiful garden",
    image=image,
    mask_image=mask,
    num_inference_steps=20,
)
```

---

## B.7 ControlNet

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import numpy as np
import cv2
from PIL import Image

# Canny边缘ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

# 准备边缘图
img = np.array(Image.open("input.jpg").resize((512, 512)))
edges = cv2.Canny(img, 100, 200)
control_image = Image.fromarray(np.stack([edges]*3, axis=-1))

result = pipe(
    prompt="a futuristic city, neon lights",
    image=control_image,
    controlnet_conditioning_scale=1.0,   # ControlNet强度（0-2）
    guidance_scale=7.5,
    num_inference_steps=20,
)

# 多ControlNet（同时使用多个条件）
# controlnets = [controlnet_canny, controlnet_depth]
# conditioning_scale = [1.0, 0.5]
```

---

## B.8 LoRA加载

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# 从HuggingFace Hub加载LoRA
pipe.load_lora_weights(
    "lora-library/some-lora-model",
    weight_name="pytorch_lora_weights.safetensors",
)

# 本地LoRA权重
pipe.load_lora_weights("./my_lora/", weight_name="adapter_model.bin")

# 设置LoRA强度
pipe.set_adapters(["lora_name"], adapter_weights=[0.8])  # 0.8强度

# 卸载LoRA
pipe.unload_lora_weights()

# 融合LoRA（加速推理，不可逆）
pipe.fuse_lora(lora_scale=1.0)
```

---

## B.9 自定义调度器（推理步骤）

```python
from diffusers import DDIMScheduler
import torch

# 手动控制采样步骤
scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)
scheduler.set_timesteps(50)  # 50步

# 手动采样循环
latents = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)

for t in scheduler.timesteps:
    # 预测噪声（需要单独加载unet和text_encoder）
    noise_pred = unet(latents, t, encoder_hidden_states=text_embeds).sample
    
    # CFG
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)
    
    # 更新
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# 解码
image = vae.decode(latents / vae.config.scaling_factor).sample
```

---

## B.10 常用工具函数

```python
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import torch

# 加载图像（支持URL、本地路径、PIL）
img = load_image("https://example.com/image.jpg")
img = load_image("./local/image.png")

# 创建图像网格（用于可视化多张图像）
images = [img1, img2, img3, img4]
grid = make_image_grid(images, rows=2, cols=2)
grid.save("grid.png")

# 图像转latent
def encode_image(vae, image, device="cuda"):
    """将PIL图像编码为VAE潜在表示"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    x = transform(image).unsqueeze(0).to(device)     # (1, 3, H, W)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample()  # (1, 4, H/8, W/8)
    return latent * vae.config.scaling_factor

# latent转图像
def decode_latent(vae, latent, device="cuda"):
    """将VAE潜在表示解码为PIL图像"""
    with torch.no_grad():
        image = vae.decode(latent / vae.config.scaling_factor).sample  # (1, 3, H, W)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((image * 255).astype('uint8'))
```

---

## B.11 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA OOM | 启用`enable_model_cpu_offload()`或减小batch size |
| 图像模糊 | 增大`num_inference_steps`或`guidance_scale` |
| 生成内容不符合提示词 | 增大`guidance_scale`（7.5→12），丰富提示词 |
| 生成速度慢 | 切换到DPM-Solver++，减少步数，启用torch.compile |
| 随机结果不可复现 | 设置`generator=torch.Generator("cuda").manual_seed(seed)` |
| 负面提示词无效 | 检查`negative_prompt`是否传入（部分API默认为空） |

---

[返回目录](../README.md)

[← 附录A：数学速查表](./math-reference.md)

[附录C：习题答案 →](./answers.md)
