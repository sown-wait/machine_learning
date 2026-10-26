"""Load MNIST and display a small labeled sample."""

import random

import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def main():
    # MNIST images are grayscale, so each pixel is represented by one value.
    transform = transforms.ToTensor()

    # torchvision 的部分默认下载地址可能无法访问，优先尝试备用镜像。
    datasets.MNIST.mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    train_data = datasets.MNIST(
        root="data",
        train=True,#60000张训练样本
        download=True,
        transform=transform,#对每张图应用上面的 ToTensor 变换
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,#10000张测试样本
        download=True,
        transform=transform,
    )

    image, label = train_data[0]#从训练集中取第 0 个样本，解包为图像张量 image（形状 [1, 28, 28]）和标签 label（整数，如 5） 
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Image shape (channels, height, width): {tuple(image.shape)}")
    print(f"Number of classes: {len(train_data.classes)}")
    print(f"Class labels: {train_data.classes}")

    sample_indices = random.sample(range(len(train_data)), 10)#随机选择 10 个样本索引
    figure, axes = plt.subplots(2, 5, figsize=(10, 5))#创建 2 行 5 列共 10 个子图，画布尺寸 10×5 英寸

    """
    axes.flat 把 2x5 的子图数组展平成一维;zip 把它和 10 个随机索引配对;
    enumerate(..., start=1) 同时得到从 1 开始的计数 sample_number。
    每次循环得到 (axis, index)：一个子图对象和一个数据索引。
    """
    for sample_number, (axis, index) in enumerate(
        zip(axes.flat, sample_indices), start=1
    ):
        image, label = train_data[index]#按索引取出图像和标签
        axis.imshow(image.squeeze(0), cmap="gray")#显示图像，去通道维度，使用灰度图映射
        axis.set_title(f"Image {sample_number} -> Label: {label}")#给子图设置标题，显示第几张图及其真实标签。
        axis.axis("off")#关闭子图的坐标轴，只显示图像。

    figure.tight_layout()#自动调整子图间距，避免标题等元素重叠
    plt.show()#显示图像


if __name__ == "__main__":
    main()
