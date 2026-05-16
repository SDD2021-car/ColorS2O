# ColorS2O：无 Color Hint 消融实验

本仓库用于运行 **SAR-to-Optical Image Translation** 的无 color hint 消融实验（ablation）。该实验不启用 color hint 相关输入或推理参数，仅使用 SAR 图像作为输入，并通过 `main_jit.py` 进行训练与生成测试。

## 实验说明

- **实验目的**：验证不使用 color hint 时模型在 SAR-to-Optical 转换任务上的表现。
- **训练脚本**：`main_jit.py`
- **默认模型**：`JiT-B/8`（如需修改，可通过 `--model` 指定）
- **图像尺寸**：`512`
- **启用损失项**：`ab perc`
- **数据目录约定**：
  - `trainA`：SAR 训练图像
  - `trainB`：Optical 训练图像
  - `testA`：SAR 测试图像

> 下面命令中的 `XX` 需要替换为实际的 GPU 编号、端口号、数据集路径、checkpoint 路径或输出路径。

## 环境准备

建议在已安装 PyTorch、torchvision、TensorBoard 等依赖的 Python 环境中运行。启动分布式训练/测试时使用 `torchrun`。

## 数据准备

请按如下结构组织数据集：

```text
XX/
├── trainA/   # SAR training images
├── trainB/   # Optical training images
└── testA/    # SAR testing images
└── testB/    # OPT testing images
```

训练时需要同时提供 `trainA` 和 `trainB`；测试生成时只需要提供 `testA`。

## 训练

使用 2 张 GPU 进行无 color hint 消融训练：

```bash
CUDA_VISIBLE_DEVICES=XX torchrun \
  --nproc_per_node=2 \
  --master-port=XX \
  main_jit.py \
  --output_dir "XX" \
  --sar_train_path "XX/trainA" \
  --opt_train_path "XX/trainB" \
  --img_size 512 \
  --enabled_losses ab perc
```

### 参数说明

- `CUDA_VISIBLE_DEVICES=XX`：指定可见 GPU，例如 `0,1`。
- `--nproc_per_node=2`：单机启动 2 个进程，对应 2 张 GPU。
- `--master-port=XX`：分布式训练端口，请选择未被占用的端口。
- `--output_dir "XX"`：训练日志与 checkpoint 保存目录。
- `--sar_train_path "XX/trainA"`：SAR 训练集目录。
- `--opt_train_path "XX/trainB"`：Optical 训练集目录。
- `--img_size 512`：输入图像尺寸。
- `--enabled_losses ab perc`：仅启用 `ab` 与 perceptual loss，用于该消融配置。

## 测试 / 生成结果

使用训练好的 checkpoint 进行生成测试，并保留输出图像：

```bash
CUDA_VISIBLE_DEVICES=XX torchrun \
  --nproc_per_node=2 \
  --master-port=XX \
  main_jit.py \
  --resume "XX" \
  --sar_test_path "XX/testA" \
  --img_size 512 \
  --enabled_losses ab perc \
  --evaluate_gen True \
  --gen_bsz 8 \
  --keep_outputs True \
  --output_dir "XX"
```

### 参数说明

- `--resume "XX"`：待测试的 checkpoint 目录或 checkpoint 路径。
- `--sar_test_path "XX/testA"`：SAR 测试集目录。
- `--evaluate_gen True`：启用生成测试模式。
- `--gen_bsz 8`：生成阶段 batch size。
- `--keep_outputs True`：保留生成结果。
- `--output_dir "XX"`：测试日志与生成结果保存目录。

## 注意事项

1. 本消融实验不需要添加 `--use_hint_infer`、`--hint_dropout_prob`、`--hint_max_ratio`、`--hint_color_thresh`、`--hint_num_regions` 或其他 color hint 相关参数。
2. 如果只使用 1 张 GPU，请将 `CUDA_VISIBLE_DEVICES` 改为单个 GPU 编号，并将 `--nproc_per_node` 改为 `1`。
3. 如果端口冲突，请更换 `--master-port` 的值。
4. `--output_dir` 建议训练和测试分别设置不同目录，避免覆盖日志或生成结果。
