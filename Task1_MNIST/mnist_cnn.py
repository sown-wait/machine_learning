"""使用卷积神经网络完成 MNIST 手写数字分类。"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TASK_DIR = Path(__file__).resolve().parent
MODEL_DIR = TASK_DIR / "model"
RESULT_DIR = TASK_DIR / "result"

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42


class MNISTCNN(nn.Module):
    """包含两个卷积层的基础 CNN。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 输入 [batch, 1, 28, 28]，输出 [batch, 32, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 池化后宽和高减半：[batch, 32, 14, 14]
            nn.MaxPool2d(kernel_size=2),
            # 第二个卷积层提取更复杂的特征。
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 再次池化：[batch, 64, 7, 7]
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            # 输出 10 个分数，分别对应数字 0～9。
            nn.Linear(128, 10),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def create_data_loaders():
    """读取数据，并创建分批送入神经网络的数据加载器。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # 将 MNIST 像素标准化，有助于神经网络稳定训练。
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return train_data, test_data, train_loader, test_loader


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    """让模型学习一遍完整训练集，并返回平均损失和准确率。"""
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        sample_count += images.size(0)

    return total_loss / sample_count, correct / sample_count


def evaluate(model, data_loader, loss_function, device):
    """在测试集上评价模型，但不更新参数。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            sample_count += images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return (
        total_loss / sample_count,
        correct / sample_count,
        np.array(all_labels),
        np.array(all_predictions),
    )


def save_training_curves(train_losses, train_accuracies, test_accuracies):
    """保存 Loss 曲线和 Accuracy 曲线。"""
    epochs = range(1, len(train_losses) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, train_losses, marker="o")
    axes[0].set_title("CNN Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_accuracies, marker="o", label="Train")
    axes[1].plot(epochs, test_accuracies, marker="s", label="Test")
    axes[1].set_title("CNN Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(RESULT_DIR / "cnn_training_curves.png", dpi=160)
    plt.close(figure)


def save_confusion_matrix(labels, predictions):
    display = ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=range(10),
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("MNIST CNN Confusion Matrix")
    display.figure_.set_size_inches(8, 7)
    display.figure_.tight_layout()
    display.figure_.savefig(RESULT_DIR / "cnn_confusion_matrix.png", dpi=160)
    plt.close(display.figure_)


def save_prediction_examples(model, test_data, device):
    """从测试集中随机抽取 10 张图片并保存预测结果。"""
    random_generator = np.random.default_rng(RANDOM_SEED)
    indices = random_generator.choice(len(test_data), size=10, replace=False)
    images = torch.stack([test_data[int(index)][0] for index in indices]).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu().numpy()

    figure, axes = plt.subplots(2, 5, figsize=(10, 5))
    for axis, index, prediction in zip(axes.flat, indices, predictions):
        raw_image = test_data.data[int(index)].numpy()
        true_label = int(test_data.targets[int(index)])
        title_color = "green" if true_label == prediction else "red"
        axis.imshow(raw_image, cmap="gray")
        axis.set_title(
            f"True: {true_label}  Pred: {int(prediction)}",
            color=title_color,
        )
        axis.axis("off")

    figure.suptitle("MNIST CNN Predictions")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "cnn_predictions.png", dpi=160)
    plt.show()


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"训练轮数: {EPOCHS}，批大小: {BATCH_SIZE}")

    train_data, test_data, train_loader, test_loader = create_data_loaders()
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")

    model = MNISTCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    start_time = perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        test_loss, test_accuracy, labels, predictions = evaluate(
            model, test_loader, loss_function, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"训练 Loss: {train_loss:.4f} | "
            f"训练 Accuracy: {train_accuracy * 100:.2f}% | "
            f"测试 Loss: {test_loss:.4f} | "
            f"测试 Accuracy: {test_accuracy * 100:.2f}%"
        )

    training_time = perf_counter() - start_time
    final_accuracy = test_accuracies[-1]
    torch.save(model.state_dict(), MODEL_DIR / "mnist_cnn.pth")

    save_training_curves(train_losses, train_accuracies, test_accuracies)
    save_confusion_matrix(labels, predictions)
    (RESULT_DIR / "cnn_metrics.txt").write_text(
        f"模型: CNN\n"
        f"训练轮数: {EPOCHS}\n"
        f"批大小: {BATCH_SIZE}\n"
        f"学习率: {LEARNING_RATE}\n"
        f"训练时间: {training_time:.2f} 秒\n"
        f"测试准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)\n",
        encoding="utf-8",
    )

    print(f"\n总训练时间: {training_time:.2f} 秒")
    print(f"最终测试准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"模型已保存到: {MODEL_DIR / 'mnist_cnn.pth'}")
    print(f"实验结果已保存到: {RESULT_DIR}")
    save_prediction_examples(model, test_data, device)


if __name__ == "__main__":
    main()
