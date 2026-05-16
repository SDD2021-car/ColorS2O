import numpy as np
import matplotlib.pyplot as plt


def plot_pure_epoch_dependent_weight_curve(total_epochs=1000, warmup_epochs=500, color='#6C8EBF', linewidth=2):
    """
    绘制一条纯净的、呈S形上升并变平的周期相关权重曲线。

    参数:
    total_epochs: 总周期数（x轴范围）。
    warmup_epochs: 预热周期数，即曲线达到最大值（1）的周期。
    color: 曲线颜色，默认为接近原图的蓝色。
    linewidth: 曲线线宽。
    """
    # 1. 生成 Epochs 轴数据 (x 轴)
    # 涵盖 0 到 total_epochs
    epochs = np.linspace(0, total_epochs, total_epochs + 1)

    # 2. 计算权重数据 (y 轴)
    # 使用余弦预热函数（Cosine Warmup）来生成S形：
    # S形段: 在 0 到 warmup_epochs 之间，从 0 平滑上升到 1
    # 平坦段: 在 warmup_epochs 之后，保持在 1
    # gamma = np.where(epochs <= warmup_epochs,
    #                 0.5 * (1 - np.cos(np.pi * epochs / warmup_epochs)),
    #                 1.0)
    #
    # 或者使用更通用的 Sigmoid 风格，需要调整参数以精确到达1。
    # 这里的余弦预热更简单且精确。

    # 替代方案（更接近 Sigmoid 形状但需要截断）：
    # sigmoid_warmup = 1 / (1 + np.exp(-10 * (epochs / warmup_epochs - 0.5)))
    # gamma = np.where(epochs <= warmup_epochs, sigmoid_warmup, 1.0)

    # 最终采用余弦预热，因为它既有 smooth S 形又精确：
    gamma = np.where(epochs <= warmup_epochs,
                     0.5 * (1 - np.cos(np.pi * epochs / warmup_epochs)),
                     1.0)

    # 3. 创建纯净图表
    fig, ax = plt.subplots(figsize=(10, 5))  # 可以根据需要调整大小

    # 4. 绘制曲线
    ax.plot(epochs, gamma, color=color, linewidth=linewidth)

    # 5. 隐藏所有坐标轴和刻度线，以满足“不需要背景和字母”的限制
    # 这将完全移除边框、刻度、刻度标签、轴标签等。
    ax.axis('off')

    # 6. 设置数据范围 (即使隐藏了，数据仍需在范围内)
    # 确保 (0,0) 的起点和 y=1 的平坦段完全显示，不被裁剪
    ax.set_xlim(0, total_epochs)
    ax.set_ylim(-0.05, 1.05)  # 给曲线一点边缘空间

    # 7. 显示结果
    plt.show()


# 运行代码
if __name__ == '__main__':
    plot_pure_epoch_dependent_weight_curve()