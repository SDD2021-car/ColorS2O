# ColorS2O: SAR-to-Optical Image Translation with Color Hints

本项目用于 **SAR 图像到光学图像（SAR-to-Optical）翻译**。推荐流程分两步：

1. 使用 `hint_mask_generator_SAR2Opt_percentage.py` 从光学图像中生成 sparse color hint。
2. 将生成的 color hint 路径传给 `main_jit.py`，进行训练或带 hint 的测试/推理。

> README 中所有路径均使用占位符表示，请替换为你自己机器上的实际路径，避免在命令中写死个人目录。

---

## 目录结构约定

建议将数据整理为成对的 SAR/Optical 文件夹：

```text
<DATA_ROOT>/
├── trainA/   # 训练集 SAR 图像
├── trainB/   # 训练集 Optical 图像，用于训练监督与生成 train color hint
├── testA/    # 测试集 SAR 图像
└── testB/    # 测试集 Optical 图像，用于测试指标计算与生成 test color hint
```

文件名应尽量一一对应，例如：

```text
trainA/0001.png
trainB/0001.png
```

---

## 环境准备

请先安装项目依赖，并确保 PyTorch、CUDA、SAM2 相关依赖可用。`hint_mask_generator_SAR2Opt_percentage.py` 会调用 SAM2，因此还需要准备：

- `<SAM2_CHECKPOINT>`：SAM2 checkpoint 文件路径，例如 `sam2.1_hiera_large.pt`。
- `<SAM2_MODEL_CFG>`：SAM2 配置文件路径，例如 `configs/sam2.1/sam2.1_hiera_l.yaml`。

---

## Step 1：生成 Color Hint

`hint_mask_generator_SAR2Opt_percentage.py` 会读取光学图像，利用 SAM2 生成区域 mask，再输出 sparse color hint。输出目录中主要包含：

```text
<HINT_OUTPUT_DIR>/
├── color_hint/          # 直接保留 hint mask 位置原始颜色，其余为黑色
├── color_hint_by_dots/  # 使用每个 dot 区域的主色填充，通常作为训练/测试 hint 输入
├── hint_masks/          # 每张图的 hint mask，npz 格式
├── label_map/           # SAM2 分割 label map
├── label_overlay/       # 分割可视化结果
└── summary.csv          # 每张图的 hint 面积比例统计
```

### 1.1 为训练集生成 hint

```bash
python hint_mask_generator_SAR2Opt_percentage.py \
  --input "<DATA_ROOT>/trainB" \
  --sam2_checkpoint "<SAM2_CHECKPOINT>" \
  --model_cfg "<SAM2_MODEL_CFG>" \
  --device cuda \
  --devices "<GPU_IDS_FOR_HINT_GENERATION>" \
  --output-dir "<TRAIN_HINT_OUTPUT_DIR>" \
  --summary-csv "<TRAIN_HINT_OUTPUT_DIR>/summary.csv"
```

示例占位符说明：

- `<DATA_ROOT>/trainB`：训练集 optical 图像路径。
- `<SAM2_CHECKPOINT>`：SAM2 权重文件路径。
- `<SAM2_MODEL_CFG>`：SAM2 模型配置文件路径。
- `<GPU_IDS_FOR_HINT_GENERATION>`：用于生成 hint 的 GPU 编号，例如 `0` 或 `0,1,2,3`。
- `<TRAIN_HINT_OUTPUT_DIR>`：训练集 hint 输出根目录。

训练时推荐将以下路径作为 `--hint_train_path`：

```text
<TRAIN_HINT_OUTPUT_DIR>/color_hint_by_dots
```

如果你想使用原始 sparse color hint，也可以改用：

```text
<TRAIN_HINT_OUTPUT_DIR>/color_hint
```

### 1.2 为测试集生成 hint

```bash
python hint_mask_generator_SAR2Opt_percentage.py \
  --input "<DATA_ROOT>/testB" \
  --sam2_checkpoint "<SAM2_CHECKPOINT>" \
  --model_cfg "<SAM2_MODEL_CFG>" \
  --device cuda \
  --devices "<GPU_IDS_FOR_HINT_GENERATION>" \
  --output-dir "<TEST_HINT_OUTPUT_DIR>" \
  --summary-csv "<TEST_HINT_OUTPUT_DIR>/summary.csv"
```

测试/推理时推荐将以下路径作为 `--hint_test_path`：

```text
<TEST_HINT_OUTPUT_DIR>/color_hint_by_dots
```

---
### 1.3 调整 hint 占比

生成 hint 的占比由 `hint_mask_generator_SAR2Opt_percentage.py` 里的 `target_ratio` 控制，默认值为 `0.05`，表示生成的 hint mask 面积目标约为整张图像面积的 **5%**。

如果想改变 color hint 的稀疏程度，可以在代码中找到 `hint_defaults` 里的这一行：

```python
target_ratio=0.05,
```

然后按需求修改：

- `target_ratio=0.01`：约 1% 像素作为 hint，更稀疏。
- `target_ratio=0.03`：约 3% 像素作为 hint。
- `target_ratio=0.10`：约 10% 像素作为 hint，更密集。

修改后重新运行 `hint_mask_generator_SAR2Opt_percentage.py` 即可生成对应占比的 color hint。脚本会在 `summary.csv` 中记录每张图实际生成的 `hint_mask_ratio`，可用于检查实际 hint 占比。

### 1.4 验证 dominance-checked rule 的效果

`hint_mask_generator_SAR2Opt_percentage.py` 还会用 `histogram_threshold` 控制 dot 区域的颜色主导性筛选（dominance check）。当前默认设置为：

```python
histogram_threshold=0.6,
```

含义是：只有当候选 dot 内最主要颜色 bin 的占比达到约 60% 时，该 dot 才会被保留为有效 hint。这样可以过滤掉颜色混杂、主导颜色不明显的 hint 点。

如果需要验证 **dominance-checked rule** 的效果，可以将该参数改为：

```python
histogram_threshold=0,
```

此时基本不再按颜色主导性过滤候选 dot，相当于关闭 dominance check。建议分别生成两组 hint 并对比：

1. `histogram_threshold=0.6`：开启 dominance check。
2. `histogram_threshold=0`：关闭 dominance check，用作 ablation。

为了公平对比，建议保持 `target_ratio`、`rng_seed`、输入数据、SAM2 checkpoint/config 和训练/测试命令不变，仅改变 `histogram_threshold`，然后比较两组 hint 的可视化结果、`summary.csv` 中的 `hint_mask_ratio` 以及最终训练/测试指标。

## Step 2：使用 Color Hint 训练 main_jit

生成训练集 color hint 后，运行 `main_jit.py` 进行训练。

```bash
CUDA_VISIBLE_DEVICES=<TRAIN_GPU_IDS> torchrun \
  --nproc_per_node=2 \
  --master-port=<MASTER_PORT> \
  main_jit.py \
  --output_dir "<OUTPUT_DIR>" \
  --resume "<RESUME_CHECKPOINT_DIR_OR_FILE>" \
  --sar_train_path "<DATA_ROOT>/trainA" \
  --opt_train_path "<DATA_ROOT>/trainB" \
  --hint_train_path "<TRAIN_HINT_OUTPUT_DIR>/color_hint_by_dots" \
  --img_size 512 \
  --hint_loss_weight 0 \
  --enabled_losses ab perc
```

参数说明：

- `<TRAIN_GPU_IDS>`：训练使用的 GPU，例如 `0,1`。需要与 `--nproc_per_node=2` 的进程数对应。
- `<MASTER_PORT>`：分布式训练端口，例如 `29501`。同一机器上多个任务不要使用相同端口。
- `<OUTPUT_DIR>`：训练日志、checkpoint、评估输出保存目录。
- `<RESUME_CHECKPOINT_DIR_OR_FILE>`：恢复训练或加载预训练权重的路径；如果从头训练，可按代码逻辑设置为空或去掉该参数。
- `<DATA_ROOT>/trainA`：训练 SAR 图像目录。
- `<DATA_ROOT>/trainB`：训练 Optical 图像目录。
- `<TRAIN_HINT_OUTPUT_DIR>/color_hint_by_dots`：Step 1 生成的训练集 color hint 目录。
- `--img_size 512`：输入图像尺寸。
- `--hint_loss_weight 0`：hint 像素额外 loss 权重，此处设置为 0。
- `--enabled_losses ab perc`：启用 Lab ab loss 和 perceptual loss。

---

## Step 3：测试 / 推理

测试前请先为 `testB` 生成测试集 color hint，然后运行：

```bash
CUDA_VISIBLE_DEVICES=<TEST_GPU_IDS> torchrun \
  --nproc_per_node=2 \
  --master-port=<MASTER_PORT> \
  main_jit.py \
  --resume "<RESUME_CHECKPOINT_DIR_OR_FILE>" \
  --output_dir "<OUTPUT_DIR>" \
  --sar_test_path "<DATA_ROOT>/testA" \
  --opt_test_path "<DATA_ROOT>/testB" \
  --hint_test_path "<TEST_HINT_OUTPUT_DIR>/color_hint_by_dots" \
  --img_size 512 \
  --hint_loss_weight 0 \
  --evaluate_gen True \
  --gen_bsz 8 \
  --keep_outputs True \
  --use_hint_infer True \
  --enabled_losses ab perc
```

参数说明：

- `<TEST_GPU_IDS>`：测试/推理使用的 GPU，例如 `0,1`。
- `<RESUME_CHECKPOINT_DIR_OR_FILE>`：待测试的 checkpoint 路径。
- `<OUTPUT_DIR>`：测试结果、生成图像和指标保存目录。
- `<DATA_ROOT>/testA`：测试 SAR 图像目录。
- `<DATA_ROOT>/testB`：测试 Optical 图像目录，用于计算指标。
- `<TEST_HINT_OUTPUT_DIR>/color_hint_by_dots`：Step 1 生成的测试集 color hint 目录。
- `--evaluate_gen True`：开启生成评估模式。
- `--gen_bsz 8`：生成阶段 batch size。
- `--keep_outputs True`：保留生成结果。
- `--use_hint_infer True`：推理阶段使用 color hint。

---

## 常用占位符速查

| 占位符 | 含义 |
| --- | --- |
| `<DATA_ROOT>` | 数据集根目录，包含 `trainA/trainB/testA/testB` |
| `<TRAIN_HINT_OUTPUT_DIR>` | 训练集 hint 输出根目录 |
| `<TEST_HINT_OUTPUT_DIR>` | 测试集 hint 输出根目录 |
| `<SAM2_CHECKPOINT>` | SAM2 checkpoint 路径 |
| `<SAM2_MODEL_CFG>` | SAM2 model config 路径 |
| `<GPU_IDS_FOR_HINT_GENERATION>` | 生成 hint 使用的 GPU，如 `0,1` |
| `<TRAIN_GPU_IDS>` | 训练使用的 GPU，如 `0,1` |
| `<TEST_GPU_IDS>` | 测试使用的 GPU，如 `0,1` |
| `<MASTER_PORT>` | torchrun 分布式端口，如 `29501` |
| `<OUTPUT_DIR>` | 训练或测试输出目录 |
| `<RESUME_CHECKPOINT_DIR_OR_FILE>` | 预训练权重、checkpoint 文件或 checkpoint 目录 |

---

## 完整流程示例（使用占位符）

```bash
# 1) 生成训练集 hint
python hint_mask_generator_SAR2Opt_percentage.py \
  --input "<DATA_ROOT>/trainB" \
  --sam2_checkpoint "<SAM2_CHECKPOINT>" \
  --model_cfg "<SAM2_MODEL_CFG>" \
  --device cuda \
  --devices "0,1" \
  --output-dir "<TRAIN_HINT_OUTPUT_DIR>" \
  --summary-csv "<TRAIN_HINT_OUTPUT_DIR>/summary.csv"

# 2) 生成测试集 hint
python hint_mask_generator_SAR2Opt_percentage.py \
  --input "<DATA_ROOT>/testB" \
  --sam2_checkpoint "<SAM2_CHECKPOINT>" \
  --model_cfg "<SAM2_MODEL_CFG>" \
  --device cuda \
  --devices "0,1" \
  --output-dir "<TEST_HINT_OUTPUT_DIR>" \
  --summary-csv "<TEST_HINT_OUTPUT_DIR>/summary.csv"

# 3) 训练
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master-port=29501 \
  main_jit.py \
  --output_dir "<OUTPUT_DIR>/train" \
  --resume "<RESUME_CHECKPOINT_DIR_OR_FILE>" \
  --sar_train_path "<DATA_ROOT>/trainA" \
  --opt_train_path "<DATA_ROOT>/trainB" \
  --hint_train_path "<TRAIN_HINT_OUTPUT_DIR>/color_hint_by_dots" \
  --img_size 512 \
  --hint_loss_weight 0 \
  --enabled_losses ab perc

# 4) 测试/推理
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master-port=29502 \
  main_jit.py \
  --resume "<RESUME_CHECKPOINT_DIR_OR_FILE>" \
  --output_dir "<OUTPUT_DIR>/test" \
  --sar_test_path "<DATA_ROOT>/testA" \
  --opt_test_path "<DATA_ROOT>/testB" \
  --hint_test_path "<TEST_HINT_OUTPUT_DIR>/color_hint_by_dots" \
  --img_size 512 \
  --hint_loss_weight 0 \
  --evaluate_gen True \
  --gen_bsz 8 \
  --keep_outputs True \
  --use_hint_infer True \
  --enabled_losses ab perc
```
