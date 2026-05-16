# ColorS2O / JiT SAR-to-Optical Translation

本仓库基于 Just image Transformer (JiT) 做 SAR 图像到光学图像的翻译/上色。训练时使用成对的 SAR 图像和光学图像；测试时从 SAR 图像生成对应的光学结果，并可选使用从光学图像或外部 hint 图像构造的颜色提示（color hints）。

## 1. 代码结构概览

| 文件/目录 | 作用 |
| --- | --- |
| `main_jit.py` | 训练和测试入口；解析命令行参数、构建数据集/模型/优化器、恢复 checkpoint，并根据 `--evaluate_gen` 决定是训练还是测试生成。 |
| `engine_jit.py` | 核心训练和测试循环；`train_one_epoch` 负责单 epoch 训练，`evaluate` 负责加载测试集、调用模型生成图片并保存结果。 |
| `denoiser.py` | 封装 JiT denoiser 模型、EMA、损失函数和 hint 相关输入；`--enabled_losses` 控制启用哪些额外图像损失。 |
| `util/datasets.py` | 数据集读取和 hint 构造；包含 SAR 单图数据集、SAR/Opt 成对数据集，以及 `build_hints`。 |
| `util/hint_vis.py` | **可选的 hint 可视化工具**；只负责把 hint 保存成图片，方便检查 hint 采样效果，不是训练/测试的必要步骤。 |
| `util/misc.py`, `util/lr_sched.py` | 分布式训练、日志、checkpoint 保存、学习率调度等工具函数。 |

## 2. 数据组织方式

推荐使用成对目录：

```text
dataset_root/
├── trainA/   # SAR 训练图像
├── trainB/   # 光学训练图像，与 trainA 文件名对应
├── testA/    # SAR 测试图像
└── testB/    # 光学测试图像，与 testA 文件名对应；启用 hint 推理时需要
```

常用参数说明：

- `--sar_train_path` / `--opt_train_path`：训练集 SAR 和光学图像目录。
- `--sar_test_path` / `--opt_test_path`：测试集 SAR 和光学图像目录。
- `--hint_train_path` / `--hint_test_path`：可选；如果已有预生成 hint 图像，可以指定路径；不指定时由代码根据光学图像动态采样 hint。
- `--img_size 512`：输入和输出图像统一 resize 到 512。

## 3. hint 与 hint_vis 的关系

### 3.1 hint 是什么

hint 是模型的颜色提示输入，由两部分拼接得到：

1. `hint_color`：提示颜色图，包含被采样位置/区域的 RGB 颜色。
2. `hint_mask`：提示掩码，标记哪些像素位置有颜色提示。

训练时，`engine_jit.py` 会把 `hint_color` 和 `hint_mask` 拼接成 `hint_input` 后传入模型。测试时，如果 `--use_hint_infer True`，也会构造并传入同样形式的 hint。

关键 hint 参数：

- `--hint_sampling_mode dot`：用点状方式采样颜色提示；也支持 `stripe`。
- `--hint_max_ratio 0.05`：hint 覆盖的最大像素比例。
- `--hint_dropout_prob 0`：训练/测试时丢弃全部 hint 的概率；设为 `0` 表示不随机丢弃。
- `--hint_loss_weight 0`：hint 像素处额外损失权重；设为 `0` 表示不额外加权。
- `--hint_on_gpu`：训练时在 GPU 上动态生成 hint，通常更方便。

### 3.2 hint_vis 是什么

`hint_vis` 只是**可选可视化功能**，用于检查采样出来的 hint 是否符合预期。它不会改变模型结构，也不是正常训练/测试必须执行的逻辑。

`util/hint_vis.py` 会保存三类图片：

- `overlay/`：把 hint 颜色覆盖到光学图像上的效果图。
- `mask/`：hint mask 的黑白可视化。
- `color/`：hint color 图本身。

### 3.3 正常训练和测试时是否需要 hint_vis

不需要。默认情况下：

- `--save_hint_vis` 默认为 `False`，测试时不会保存 hint 可视化。
- `--visualize_hints_only` 默认为 `False`，程序不会进入“只保存 hint 后退出”的模式。

因此，**直接使用下面的训练命令和测试命令即可**，不用传任何 `hint_vis` 相关参数。

### 3.4 什么时候才打开 hint_vis

只有在你想检查 hint 采样是否正确时才需要打开：

- 测试生成时同时保存 hint 可视化：加 `--save_hint_vis True`，并保持 `--use_hint_infer True`。
- 只生成 hint 可视化、不训练也不测试：加 `--visualize_hints_only True`，可配合 `--hint_vis_dir` 和 `--hint_vis_max_images` 指定保存位置和数量。

如果只是正常训练和测试，请不要加这些参数。

## 4. 训练命令

将下面命令中的 `XX` 替换成自己的 GPU 编号、端口、输出目录和数据集路径。

```bash
CUDA_VISIBLE_DEVICES=XX torchrun --nproc_per_node=2 --master-port=XX main_jit.py \
  --output_dir "XX" \
  --sar_train_path "XX/trainA" \
  --opt_train_path "XX/trainB" \
  --img_size 512 \
  --hint_dropout_prob 0 \
  --hint_loss_weight 0 \
  --hint_sampling_mode dot \
  --hint_on_gpu \
  --hint_max_ratio 0.05 \
  --enabled_losses ab perc
```

说明：

- `CUDA_VISIBLE_DEVICES=XX`：填写要使用的 GPU，例如 `0,1`。
- `--nproc_per_node=2`：启动 2 个进程，通常对应 2 张 GPU。
- `--master-port=XX`：填写未被占用的端口，例如 `29501`。
- `--enabled_losses ab perc`：只启用 Lab ab 损失和 perceptual 损失。
- 该命令不会触发 `hint_vis`，会直接进入训练。

## 5. 测试/生成命令

将下面命令中的 `XX` 替换成自己的 checkpoint 目录、输出目录、测试集路径、GPU 编号和端口。

```bash
CUDA_VISIBLE_DEVICES=XX torchrun --nproc_per_node=2 --master-port=XX main_jit.py \
  --resume "XX" \
  --output_dir "XX" \
  --sar_test_path "XX/testA" \
  --opt_test_path "XX/testB" \
  --img_size 512 \
  --hint_dropout_prob 0 \
  --hint_loss_weight 0 \
  --hint_sampling_mode dot \
  --hint_on_gpu \
  --evaluate_gen True \
  --gen_bsz 8 \
  --keep_outputs True \
  --use_hint_infer True
```

说明：

- `--resume "XX"`：填写 checkpoint 目录，代码会读取其中的 `checkpoint-last.pth`。
- `--evaluate_gen True`：进入测试/生成模式，不进行训练。
- `--output_dir "XX"`：生成结果保存目录。
- `--use_hint_infer True`：测试时使用 hint 输入；因此需要提供 `--opt_test_path`，用于构造或读取 hint。
- `--gen_bsz 8`：生成 batch size。
- `--keep_outputs True`：保留生成图片，不在计算指标后删除输出目录。
- 该命令没有传 `--save_hint_vis`，所以只保存生成结果，不保存 hint 可视化图片。

## 6. 常见使用建议

1. **只想训练/测试：** 使用第 4、5 节命令即可，不要传 `--save_hint_vis` 或 `--visualize_hints_only`。
2. **想看 hint 采样效果：** 在测试命令中额外加 `--save_hint_vis True`，输出目录下会多出 `hints/overlay`、`hints/mask`、`hints/color`。
3. **想关闭测试 hint：** 可以把 `--use_hint_infer True` 改为 `--use_hint_infer False`，此时只使用 SAR 图像生成，不读取测试光学图像来构造 hint。
4. **多 GPU 训练/测试：** `CUDA_VISIBLE_DEVICES` 中 GPU 数量应与 `--nproc_per_node` 一致，例如 `CUDA_VISIBLE_DEVICES=0,1` 搭配 `--nproc_per_node=2`。
