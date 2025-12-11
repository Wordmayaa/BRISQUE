# BRISQUE

# 核心依赖
pip install torch torchvision numpy
# 安装 piq 库
pip install piq

import torch
import numpy as np
import piq
from piq import brisque

def calculate_brisque_piq(image_tensor: torch.Tensor) -> float:
    """
    使用 piq 库计算图像张量的 BRISQUE 分数。

    参数:
        image_tensor (torch.Tensor): 
            输入图像张量。要求格式为 (C, H, W) 或 (N, C, H, W)，
            且像素值范围应为 [0, 1] 或 [0, 255]。
            BRISQUE通常在灰度图上计算，所以 C=1 或 C=3（内部会转灰度）。
        
    返回:
        float: BRISQUE 分数。
    """
    # 1. 确保张量在 CPU 上且格式正确（如果它是在 GPU 上）
    image_tensor = image_tensor.cpu().float()
    
    # 2. BRISQUE 模型权重加载
    # BRISQUE 需要加载预训练的 SVR 模型参数。
    # piq 库会自动处理此步骤。
    
    # 3. 计算 BRISQUE 分数
    # is_data_a_batch=False 适用于 (C, H, W) 格式的单张图像
    # data_range=255. 表示输入图像像素值范围是 [0, 255]
    try:
        score_tensor = brisque(
            image_tensor, 
            data_range=255., 
            reduction='none', 
            data_format='CHW'
        )
        
        score = score_tensor.item()
        
        print("-" * 30)
        print(f"BRISQUE 分数: {score:.4f}")
        print("-" * 30)
        
        return score

    except Exception as e:
        print(f"BRISQUE 计算出错: {e}")
        return None

# --- 示例用法 ---

# 假设您有一张 256x256 的灰度图像（单通道 C=1）
H, W = 256, 256
# 1. 创建一个随机的单张灰度图 PyTorch 张量 (C=1, H, W)
# 像素值范围设定为 [0, 255]
dummy_image_data = np.random.randint(0, 256, (1, H, W), dtype=np.uint8)
dummy_tensor = torch.from_numpy(dummy_image_data).float()

print(f"--- 正在使用 PyTorch 张量进行测试 ---")
print(f"输入张量形状: {dummy_tensor.shape}, 值域: [0.0, 255.0]")

# 计算分数
brisque_score = calculate_brisque_piq(dummy_tensor)

# 🎯 记住: BRISQUE 分数越**小**，图像感知质量越**好**。
