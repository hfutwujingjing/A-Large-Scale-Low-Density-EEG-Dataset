# baseline1000_HBN_MODMA

基于 **HBN + MODMA 两个公开 EEG 数据集**的跨数据集蒸馏基线，最终在自采数据（Ours）上微调分类。

核心思路：利用 128 通道数据训练"教师模型"，再通过**知识蒸馏**把知识压缩到 3 通道（额部前区导联）的"学生模型"，最后用学生模型拼接成双分支结构在自采数据上微调，实现低成本通道数下的 EEG 分类。

- 名称中 `1000` 表示输入序列长度为 1000（`SEQ_LEN = 1000`）。
- 数据为预切分好的 segment（npz 格式），每个样本为 `[通道数, 时间点数]`，默认 128 通道截断为前 128 个导联，2000 点下采样到 1000 点（隔点采样）。

## 训练流程（4 阶段）

整个流程由 [main.py](main.py) 串行驱动，共 4 个阶段：

```
Stage 1a: HBN 教师预训练（128ch 重建）
   └─ 自监督重建：Encoder + 解码器把 patch 重建回原始波形，MSE 损失
          ↓ 得到 hbn_teacher_128.pth
Stage 1b: HBN 蒸馏（128ch → 3ch）
   └─ 用 128ch 教师输出特征（stop-grad）监督 3ch 学生，MSE 损失
          ↓ 得到 hbn_student_3.pth（通用/领域知识编码器）
Stage 2a: MODMA 教师（128ch 分类）
   └─ 在 MODMA 上训练 128ch 分类器（rest + erp），CrossEntropy，按 val F1 存档
          ↓ 得到 modma_teacher_128.pth
Stage 2b: MODMA 蒸馏（128ch → 3ch）
   └─ 128ch 教师特征蒸馏到 3ch 学生
          ↓ 得到 modma_student_3.pth（类别/任务知识编码器）
Stage 3: 自采数据（Ours）微调
   └─ DualBranchModel：拼接 [HBN 通用编码器特征, MODMA 分类编码器特征] 后接 MLP 分类
          ↓ 按 val F1 存 hbn_modma_best.pth，最终在 Ours eval 集上测试
```

阶段与超参对应关系（见 [config.py](config.py)）：

| 阶段 | 脚本 | 关键超参 |
|------|------|----------|
| 1a HBN 重建预训练 | [train_hbn.py](train_hbn.py) | `HBN_PRETRAIN_*`（50 epochs, lr 1e-4, batch 64） |
| 1b HBN 蒸馏 | [distill.py](distill.py) | `HBN_DISTILL_*`（50 epochs, lr 5e-5, batch 128） |
| 2a MODMA 教师分类 | [train_modma.py](train_modma.py) | `MODMA_TEACHER_*`（50 epochs, lr 1e-4, batch 64） |
| 2b MODMA 蒸馏 | [distill.py](distill.py) | `MODMA_DISTILL_*`（50 epochs, lr 5e-5, batch 128） |
| 3 Ours 微调 | [train.py](train.py) + [test.py](test.py) | `EPOCHS=100, BATCH_SIZE=32, LR=1e-4` |

## 目录结构

```
baseline1000_HBN_MODMA/
├── config.py          # 数据路径 + 全部超参数 + 设备选择
├── data_loader.py     # 各阶段 Dataset / DataLoader 构建
├── model.py           # EEGEncoder、PretrainModel、DistillModel、ClassifierModel、DualBranchModel
├── train_hbn.py       # Stage 1a：重建预训练（MSE）
├── train_modma.py     # Stage 2a：MODMA 分类教师（CE，按 val F1 存档）
├── distill.py         # Stage 1b/2b：特征蒸馏（MSE）
├── train.py           # Stage 3：双分支微调（CE + CosineAnnealing，按 val F1 存档）
├── test.py            # 测试集评估（Acc / Pre / Rec / F1）
└── main.py            # 端到端入口，按顺序跑完 4 个阶段
```

## 模型结构

[model.py](model.py) 的核心是 **Patch-Embedding + Transformer 的 EEG Encoder**：

- **PatchEmbedding**：`Conv1d` 把每段信号切成 `SEQ_LEN / PATCH_SIZE = 20` 个 patch，映射到 `d_model=128`。
- **PositionalEncoding**：标准正弦位置编码。
- **EEGEncoder**：CLS token + Transformer Encoder（`NHEAD=4`，`NUM_LAYERS=3`，前归一化，GELU），取 CLS 位置输出作为整段信号特征。
- **PretrainModel**：Encoder 输出 patch 经解码器重建回原始波形（自监督）。
- **DistillModel**：教师（冻结）与学生双输入，输出 `(学生特征, 教师特征)`。
- **ClassifierModel**：Encoder + 线性分类头，返回 `(logits, 特征)`。
- **DualBranchModel**：两个编码器特征拼接 → `Linear(256,128) → GELU → Dropout → Linear(128,3)` 分类，支持分别冻结通用/分类分支。

## 数据要求

各数据集路径硬编码在 [config.py](config.py)（`/16T/wxp/dataset/OSF_MODMA_Ours/` 下），使用前需按实际位置修改：

| 数据集 | 目录 | 需要的 npz 文件 |
|--------|------|-----------------|
| HBN | `HBN_split` | `HBN_rest_segments.npz`、`HBN_stim_segments.npz`、`HBN_{rest,stim}_fp3_segments.npz` |
| MODMA | `MODMA_split` | `MODMA_rest_128ch_segments.npz`、`MODMA_erp_128ch_segments.npz`、`MODMA_3ch_segments.npz`（可选） |
| Ours | `Ours_split` 下 `Round4_{train,eval}`、`Guangyuan_{train,eval}` | `Ours_still_segments.npz`、`Ours_sound_segments.npz` |

npz 格式：每个文件内以被试 id 为 key，`data[id].item()` 返回含 `"segments"`（`[n_seg, n_ch, n_time]`，float32）的 dict；分类数据集还需 `"label"` 字段。

**3 通道选取**：蒸馏目标通道由 `FP3_IDX_HBN = [21, 15, 8]`、`FP3_IDX_MODMA = [21, 10, 8]` 指定（额部前区附近导联，不同 10-20 系统编号不同）。

## 运行

```bash
# 安装依赖
pip install torch numpy scikit-learn tqdm

# 修改 config.py 中的数据集路径后，端到端跑完整流程
python main.py

# 或分阶段单独训练（需先保证前置阶段产物存在）
# 例如只做 Stage 3 微调时，先手动加载 1b/2b 阶段保存的 3ch 学生权重
```

模型权重默认保存到 `./pth/`：

```
pth/
├── hbn_teacher_128.pth      # 1a 产物
├── hbn_student_3.pth        # 1b 产物
├── modma_teacher_128.pth    # 2a 产物
├── modma_student_3.pth      # 2b 产物
└── hbn_modma_best.pth       # 3 微调最优权重
```

## 说明与注意事项

- **设备**：`DEVICE = cuda:1`（[config.py](config.py)），多卡环境请按需修改。
- **数据量**：各 Dataset 在构造时会一次性把所有 segment 读入内存，128 通道 + 128ch 数据集显存/内存占用较大。
- **冻结选项**：Stage 3 可通过 `FREEZE_GENERAL` / `FREEZE_CLASS` 冻结对应编码器分支（默认不冻结，全参数微调）。
- **MODMA 3ch 蒸馏数据增强**：`MODMADistillDataset` 中，若存在真实的 `MODMA_3ch_segments.npz`，会额外用 128ch segment 与真实 3ch segment 循环配对加入训练。
