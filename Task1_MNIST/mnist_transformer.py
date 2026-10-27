"""使用简化版 Vision Transformer 完成 MNIST 分类。"""

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
PATCH_SIZE = 4
EMBED_DIM = 64
NUM_HEADS = 4


class MNISTVisionTransformer(nn.Module):
    """把图像切成小块，再用 Transformer Encoder 分类。"""

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBED_DIM)
        # 28x28 图像切成 4x4 小块，每边 7 块，共 49 个 token。
        self.position_embedding = nn.Parameter(torch.zeros(1, 49, EMBED_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 10),
        )

    def forward(self, images):
        # images: [batch, 1, 28, 28]
        patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(
            3, PATCH_SIZE, PATCH_SIZE
        )
        # [batch, 1, 7, 7, 4, 4] -> [batch, 49, 16]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(images.size(0), 49, -1)
        tokens = self.patch_embedding(patches) + self.position_embedding
        encoded_tokens = self.encoder(tokens)
        # 平均汇总 49 个图像块的特征，再输出 10 个类别分数。
        image_features = encoded_tokens.mean(dim=1)
        return self.classifier(image_features)


def create_data_loaders():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return train_data, test_data, train_loader, test_loader


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
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
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
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
    epochs = range(1, len(train_losses) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_losses, marker="o")
    axes[0].set_title("Transformer Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_accuracies, marker="o", label="Train")
    axes[1].plot(epochs, test_accuracies, marker="s", label="Test")
    axes[1].set_title("Transformer Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.8, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(RESULT_DIR / "transformer_training_curves.png", dpi=160)
    plt.close(figure)


def save_confusion_matrix(labels, predictions):
    display = ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=range(10),
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("MNIST Transformer Confusion Matrix")
    display.figure_.set_size_inches(8, 7)
    display.figure_.tight_layout()
    display.figure_.savefig(RESULT_DIR / "transformer_confusion_matrix.png", dpi=160)
    plt.close(display.figure_)


def save_prediction_examples(model, test_data, device):
    random_generator = np.random.default_rng(RANDOM_SEED)
    indices = random_generator.choice(len(test_data), size=10, replace=False)
    images = torch.stack([test_data[int(index)][0] for index in indices]).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu().numpy()

    figure, axes = plt.subplots(2, 5, figsize=(10, 5))
    for axis, index, prediction in zip(axes.flat, indices, predictions):
        true_label = int(test_data.targets[int(index)])
        axis.imshow(test_data.data[int(index)].numpy(), cmap="gray")
        axis.set_title(
            f"True: {true_label}  Pred: {int(prediction)}",
            color="green" if true_label == prediction else "red",
        )
        axis.axis("off")

    figure.suptitle("MNIST Transformer Predictions")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "transformer_predictions.png", dpi=160)
    plt.show()


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"训练轮数: {EPOCHS}，批大小: {BATCH_SIZE}")
    print(
        f"每张 28x28 图片切为 49 个 {PATCH_SIZE}x{PATCH_SIZE} 小块，"
        f"Embedding 维度: {EMBED_DIM}"
    )

    train_data, test_data, train_loader, test_loader = create_data_loaders()
    print(f"训练集样本数: {len(train_data)}，测试集样本数: {len(test_data)}")
    model = MNISTVisionTransformer().to(device)
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
    torch.save(model.state_dict(), MODEL_DIR / "mnist_transformer.pth")
    save_training_curves(train_losses, train_accuracies, test_accuracies)
    save_confusion_matrix(labels, predictions)
    (RESULT_DIR / "transformer_metrics.txt").write_text(
        f"模型: Vision Transformer\n"
        f"训练轮数: {EPOCHS}\n"
        f"批大小: {BATCH_SIZE}\n"
        f"学习率: {LEARNING_RATE}\n"
        f"训练时间: {training_time:.2f} 秒\n"
        f"测试准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)\n",
        encoding="utf-8",
    )
    print(f"\n总训练时间: {training_time:.2f} 秒")
    print(f"最终测试准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"模型已保存到: {MODEL_DIR / 'mnist_transformer.pth'}")
    print(f"实验结果已保存到: {RESULT_DIR}")
    save_prediction_examples(model, test_data, device)


if __name__ == "__main__":
    main()
