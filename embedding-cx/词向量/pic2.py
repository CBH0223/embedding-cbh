from PIL import Image
import torchvision.transforms as transforms
import torch

# 读取图像
image_path = '/Users/chenxin/近期工作/Embedding/yuan.jpg'  # 替换成您自己的图像路径
image = Image.open(image_path)
# 定义变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])
# 应用变换
rgb_image_tensor = transform(image)
# 显示三通道图像张量
print(rgb_image_tensor)