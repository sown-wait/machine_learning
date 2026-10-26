"""使用随机森林完成 MNIST 手写数字分类。"""

from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torchvision import datasets


# 无论从哪个目录启动脚本，都把数据和结果放到固定位置。
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TASK_DIR = Path(__file__).resolve().parent
MODEL_DIR = TASK_DIR / "model"
RESULT_DIR = TASK_DIR / "result"


def load_and_preprocess_data():
    """读取 MNIST，并把二维图像转换成随机森林需要的一维特征。"""
    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True)

    # 原始形状是 [样本数, 28, 28]，展平后每张图片有 784 个特征。
    # 除以 255 后，像素值从 0～255 归一化到 0～1。
    x_train = train_data.data.numpy().reshape(len(train_data), -1).astype(np.float32) / 255.0
    x_test = test_data.data.numpy().reshape(len(test_data), -1).astype(np.float32) / 255.0
    y_train = train_data.targets.numpy()
    y_test = test_data.targets.numpy()

    print(f"训练集输入形状: {x_train.shape}")
    print(f"训练集标签形状: {y_train.shape}")
    print(f"测试集输入形状: {x_test.shape}")
    print(f"归一化后的像素范围: {x_train.min():.1f}～{x_train.max():.1f}")
    return train_data, test_data, x_train, y_train, x_test, y_test


def save_confusion_matrix(y_test, predictions):
    """绘制并保存 10 类数字的混淆矩阵。"""
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=range(10),
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("MNIST Random Forest Confusion Matrix")
    display.figure_.set_size_inches(8, 7)
    display.figure_.tight_layout()
    display.figure_.savefig(RESULT_DIR / "random_forest_confusion_matrix.png", dpi=160)
    plt.close(display.figure_)


def save_prediction_examples(test_data, predictions):
    """随机展示并保存 10 张测试图片的真实标签与预测标签。"""
    random_generator = np.random.default_rng(42)
    test_labels = test_data.targets.numpy()
    # 每个真实类别随机选择一张，确保预测示例完整覆盖数字 0～9。
    indices = [
        random_generator.choice(np.flatnonzero(test_labels == digit))
        for digit in range(10)
    ]
    figure, axes = plt.subplots(2, 5, figsize=(10, 5))

    for axis, index in zip(axes.flat, indices):
        true_label = int(test_data.targets[index])
        predicted_label = int(predictions[index])
        title_color = "green" if true_label == predicted_label else "red"
        axis.imshow(test_data.data[index].numpy(), cmap="gray")
        axis.set_title(
            f"True: {true_label}  Pred: {predicted_label}",
            color=title_color,
        )
        axis.axis("off")

    figure.suptitle("MNIST Random Forest Predictions")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "random_forest_predictions.png", dpi=160)
    plt.show()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_data, test_data, x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # 100 棵决策树分别判断，最后由多数投票决定数字类别。
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    print("\n开始训练随机森林，请耐心等待……")
    start_time = perf_counter()
    model.fit(x_train, y_train)
    training_time = perf_counter() - start_time

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"测试准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    joblib.dump(model, MODEL_DIR / "random_forest.joblib")
    (RESULT_DIR / "random_forest_metrics.txt").write_text(
        f"模型: Random Forest\n"
        f"训练样本数: {len(train_data)}\n"
        f"测试样本数: {len(test_data)}\n"
        f"训练时间: {training_time:.2f} 秒\n"
        f"测试准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)\n",
        encoding="utf-8",
    )
    save_confusion_matrix(y_test, predictions)
    save_prediction_examples(test_data, predictions)

    print(f"\n模型已保存到: {MODEL_DIR}")
    print(f"实验结果已保存到: {RESULT_DIR}")


if __name__ == "__main__":
    main()
