# BRISQUE

# 安装用于图像处理和评估的库
pip install opencv-python numpy scipy scikit-learn
# 如果使用 niqe 库
pip install niqe
# 如果使用 iqa-pytorch 库（它也包含了 BRISQUE）
# pip install iqa-pytorch

import cv2
import numpy as np
# 假设您使用的库是 niqe，它包含 brisque 实现
# 实际项目中，您可能需要查找您的特定 IQA 库中 BRISQUE 的导入路径
from niqe.brisque import calculate_brisque

def get_brisque_score(image_path):
    """
    计算给定路径图像的 BRISQUE 分数。

    参数:
        image_path (str): 图像文件的路径。
        
    返回:
        float: BRISQUE 分数。
    """
    try:
        # 1. 加载图像
        # BRISQUE通常在灰度图上计算，使用 cv2.IMREAD_GRAYSCALE 加载灰度图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"错误: 无法加载图像 '{image_path}'。")
            return None
        
        # 将图像数据类型转换为 float64，这是科学计算库常见的输入要求
        img_float = img.astype(np.float64)
        
        print(f"成功加载图像，尺寸: {img.shape}")
        
        # 2. 计算 BRISQUE 分数
        # calculate_brisque 函数接收一个 NumPy 数组作为输入
        score = calculate_brisque(img_float)
        
        # 3. 输出结果
        print("-" * 30)
        print(f"图像路径: {image_path}")
        print(f"BRISQUE 分数: {score:.4f}")
        print("-" * 30)
        
        return score

    except ImportError:
        print("错误: 请确保已安装 'niqe' 库。如果使用其他 IQA 库，请修改导入语句。")
        return None
    except Exception as e:
        print(f"发生其他错误: {e}")
        return None

# --- 示例用法 ---
# ⚠️ 注意：您需要将 'your_image.jpg' 替换为您电脑上存在的图像路径。
# 建议使用一张失真（如 JPEG 压缩、模糊）或恢复后的图像来测试。
image_file = "path/to/your_restored_image.png"

# 为了运行示例，我们先创建一个虚拟的灰度图像
try:
    # 创建一个 256x256 的随机灰度图像
    dummy_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    dummy_path = "dummy_test_image.png"
    cv2.imwrite(dummy_path, dummy_img)
    
    print("--- 正在使用虚拟图像进行测试 ---")
    get_brisque_score(dummy_path)
    
except Exception as e:
    print(f"无法创建虚拟图像或运行示例: {e}")


# 🎯 记住: BRISQUE 分数越**小**，图像感知质量越**好**。
