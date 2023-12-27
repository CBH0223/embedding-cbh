from PIL import Image
import torchvision.transforms as transforms
import torch

# 读取图像
image_path = '/Users/chenxin/近期工作/Embedding/yuan.jpg'  # 替换成您自己的图像路径
image = Image.open(image_path)

# 定义变换
transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图
    transforms.ToTensor(),   # 转换为张量
])

# 应用变换
gray_image_tensor = transform(image)

# 显示灰度图张量
print(gray_image_tensor)

# 将张量转换回PIL图像
pil_image = transforms.ToPILImage()(gray_image_tensor)

# 显示灰度图
pil_image.show()