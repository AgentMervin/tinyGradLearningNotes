# All-Reduce 笔记

> CS336 · Large-Scale Distributed Training 配套笔记
> 主题：理解 All-Reduce —— 分布式训练里最高频的集合通信原语

---

## 1. 一句话定义

> **All-Reduce = 把所有 GPU 上的同一个 tensor 做一次"聚合运算"（通常是求和），然后把结果广播回每个 GPU，让大家最终拿到完全相同的结果。**

```
Reduce     = 多 → 一       （汇总到一个节点）
Broadcast  = 一 → 多       （从一个节点发给所有人）
All-Reduce = Reduce + Broadcast = 多 → 多（人人都拿到同样的汇总结果）
```

支持的 reduce 运算：`SUM`（最常用）、`AVG`、`MAX`、`MIN`、`PROD` 等。

---

## 2. 直观图示：4 个 GPU 做 All-Reduce(sum)

### 起始状态
```
GPU0: [1, 2]      GPU1: [3, 4]      GPU2: [5, 6]      GPU3: [7, 8]
```

### All-Reduce(sum) 之后
```
GPU0: [16, 20]    GPU1: [16, 20]    GPU2: [16, 20]    GPU3: [16, 20]
       ↑
       1+3+5+7=16, 2+4+6+8=20，每张卡都拿到一样的结果
```

---

## 3. 在分布式训练里的两大经典场景

### 3.1 Data Parallel —— 同步梯度

```
每张卡用自己的 mini-batch 算出 ∇L_i
              │
              ▼
   All-Reduce(sum) on gradients
              │
              ▼
每张卡都拿到 Σ∇L_i  ⇒  本地除以 N ⇒ 用同样的梯度更新参数
                    （保证所有卡的模型参数始终一致）
```

- **频率**：每个 training step 一次
- **通信量**：≈ 2 × 模型参数量（见 §5）
- **是 DP 训练的性能瓶颈之一**

### 3.2 Tensor Parallel (Megatron) —— 合并被切开的 GEMM

以 FFN 的两次矩阵乘 `Z = (X·W₁)·W₂` 为例，把 `W₁` 按列切、`W₂` 按行切到 2 张卡：

```
GPU0:  Y0 = X · W₁[:, :H/2]        GPU1:  Y1 = X · W₁[:, H/2:]
GPU0:  Z0 = Y0 · W₂[:H/2, :]        GPU1:  Z1 = Y1 · W₂[H/2:, :]
                  │                              │
                  └──────── All-Reduce(sum) ─────┘
                              │
                       每卡都拿到完整 Z = Z0 + Z1
```

- **频率**：每层 forward 1 次 + backward 1 次
- 这就是 Megatron-LM 中 attention 和 MLP 块末尾的那个 all-reduce

---

## 4. Ring All-Reduce：为什么不慢死？

朴素做法：所有人发给 0 号 → 0 号求和 → 再广播。0 号带宽爆炸，**O(N)** 不可扩展。

**NCCL 默认用 Ring All-Reduce**，把 N 张卡组成一个环：

```
       GPU0 ──► GPU1
        ▲         │
        │         ▼
       GPU3 ◄── GPU2
```

把待聚合的 tensor（大小 D）切成 **N 块**，分两个阶段，每阶段 N-1 步：

| 阶段 | 做什么 | 单卡通信量 |
|---|---|---|
| **Reduce-Scatter** | 沿环传递并累加，最终每张卡持有 1 块的"全局和" | (N-1)/N · D |
| **All-Gather** | 每张卡把自己那 1 块沿环传一圈，凑齐完整结果 | (N-1)/N · D |
| **总计** | | **≈ 2D**，与 N 几乎无关 ✅ |

> 🔑 **Ring All-Reduce 的精髓**：通信量与 GPU 数 N 解耦，这是大规模训练能扩展到几千卡的基础之一。

### Reduce-Scatter 一步步走（N=4，每卡持 4 个分块 a/b/c/d）

```
初始：
  GPU0: a0 b0 c0 d0
  GPU1: a1 b1 c1 d1
  GPU2: a2 b2 c2 d2
  GPU3: a3 b3 c3 d3

每一步：第 i 张卡把自己负责的那块发给下一张卡，下一张卡累加。
3 步后：
  GPU0: ?  ?  ?  Σd     ← 持有 d 的全局和
  GPU1: Σa ?  ?  ?      ← 持有 a 的全局和
  GPU2: ?  Σb ?  ?      ← 持有 b 的全局和
  GPU3: ?  ?  Σc ?      ← 持有 c 的全局和
```

接着 All-Gather 再走 3 步，每张卡把自己手里的那块全局和传一圈，最终人人都凑齐 `[Σa, Σb, Σc, Σd]`。

---

## 5. 通信量计算（重要）

| 算法 | 单卡发送+接收总量 | 与 N 关系 |
|---|---|---|
| 朴素 Reduce + Broadcast | 2(N-1)·D / N ≈ 2D（root 节点 2N·D 爆炸） | root 是瓶颈 |
| **Ring All-Reduce** | **2·(N-1)/N · D ≈ 2D** | 与 N 无关 ✅ |
| Tree All-Reduce（NCCL 也支持） | ≈ 2D，延迟 O(log N) | 大 N 下延迟更优 |

> 训练中常说"DP 每 step 通信 ≈ 2 × 参数量"，就是这么来的。
> 例如 70B 模型 fp16 梯度 = 140GB，每 step 单卡要走 ~280GB 通信量。

---

## 6. 集合通信全家福对比

| 集合通信 | 输入 | 输出 | 典型用途 |
|---|---|---|---|
| **Broadcast** | 1 张卡有 X | 所有卡都有 X | 初始化参数广播 |
| **Reduce** | 每卡有 X_i | 1 张卡有 Σ X_i | 不常单用 |
| **All-Reduce** | 每卡有 X_i | 每卡都有 Σ X_i | **DP 梯度同步、TP 合并** |
| **Reduce-Scatter** | 每卡有 X_i (大) | 每卡有 Σ 的一部分 | **ZeRO、FSDP 梯度切片** |
| **All-Gather** | 每卡有一部分 | 每卡有完整拼接 | **FSDP forward 还原权重** |
| **All-to-All** | 每卡发不同片给不同人 | 每卡收齐 | **MoE expert routing、Sequence Parallel** |

> 💡 **关键恒等式**：
> ```
> All-Reduce  ≡  Reduce-Scatter  +  All-Gather
> ```
> 这就是为什么 **ZeRO / FSDP** 可以把"同步梯度的 all-reduce"拆成两步：
> - **Reduce-Scatter**：同步梯度时，每卡只保留自己那一片（梯度被切了 → 省显存）
> - **All-Gather**：forward/backward 用到完整权重时，再临时凑齐
>
> 总通信量不变（还是 2D），但 **显存** 从 O(D) 降到 O(D/N)。

---

## 7. PyTorch 代码示例

### 7.1 基础 all_reduce
```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

x = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)
# 每张卡的 x 都变成 1+2+...+world_size
print(f"rank {rank}: {x.item()}")
```

### 7.2 DDP 中手动同步梯度（DDP 内部就是这么做的）
```python
for p in model.parameters():
    if p.grad is not None:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
```

### 7.3 Reduce-Scatter + All-Gather 等价于 All-Reduce
```python
# 假设 grad 大小 D，world_size=N，D 能被 N 整除
shard = torch.empty(D // N, device=device)
dist.reduce_scatter_tensor(shard, grad, op=dist.ReduceOp.SUM)
# 此时每卡只有 grad 的一片全局和（FSDP 在这步停下，省显存）

dist.all_gather_into_tensor(grad, shard)
# 凑齐后等价于一次 all_reduce
```

---

## 8. 工程优化要点

1. **Overlap 计算与通信**
   - DDP 在 backward 时，**算完一层的梯度就立刻 all-reduce**，与下一层的 backward 计算并行。
   - 这是 PyTorch DDP 默认行为（"gradient bucketing"）。

2. **Bucketing（梯度分桶）**
   - 把多个小 tensor 拼成大 tensor 一次 all-reduce，减少 launch 开销。
   - 桶大小（默认 25MB）影响 overlap 效果。

3. **拓扑感知**
   - **节点内**走 NVLink（~600 GB/s），**节点间**走 InfiniBand/RoCE（~50–400 GB/s）。
   - NCCL 会自动构建分层 ring（节点内 ring + 节点间 ring）。

4. **精度**
   - fp16/bf16 梯度直接 all-reduce 即可。
   - fp8 训练里通常仍用 bf16/fp32 做 all-reduce 保证数值稳定。

5. **压缩 / 量化通信**
   - PowerSGD、1-bit Adam 等：用低秩/量化压缩梯度后再通信，但有精度风险。

---

## 9. CS336 视角的核心 takeaways

1. **All-Reduce 是 DP 训练每 step 必做的操作**，通信量 ≈ 2 × 参数量。
2. **Ring 算法让通信量与 GPU 数无关**，是 scale 到几千卡的基础。
3. **All-Reduce ≡ Reduce-Scatter + All-Gather**，这是 ZeRO/FSDP 省显存的数学基础。
4. **TP 中的 all-reduce 出现在每层 attention/MLP 块末尾**，比 DP 频率高得多 → 必须放在 NVLink 域内（节点内）。
5. **优化的关键是 overlap**：让 all-reduce 和反向计算并行，把通信"藏"在计算下面。

---

## 10. 延伸阅读

- NCCL 文档：<https://docs.nvidia.com/deeplearning/nccl/>
- "Bringing HPC Techniques to Deep Learning" (Baidu Ring All-Reduce 原论文)
- ZeRO 论文：Rajbhandari et al., 2020
- Megatron-LM 论文：Shoeybi et al., 2019
- PyTorch DDP 论文：Li et al., VLDB 2020
