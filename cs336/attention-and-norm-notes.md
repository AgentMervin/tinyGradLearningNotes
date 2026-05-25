# Attention & Normalization — 公式优先速通

> 📅 整理时间：2026-05-19
> 🎯 场景：今天对话的全部干货整理。**先看公式，再看解释**。
> 🧭 阅读顺序：① 形状约定 → ② Single-Head Attention → ③ Multi-Head Attention → ④ LayerNorm → ⑤ LayerNorm vs BatchNorm
> 🔗 配套：[`transformer-primer.md`](transformer-primer.md)

---

## 0. 形状约定（贯穿全文）⭐ 先记住

| 符号 | 含义 | 典型值 |
|---|---|---|
| `B` | batch size | 32 |
| `T` | sequence length（token 数）| 1024 |
| `C = d = d_model` | **每个 token 的维度**（特征维 / model dim / hidden size / embedding dim — 同一个数） | 768 |
| `h` | 多头数 | 12 |
| `d_k = C / h` | 每个 head 的维度 | 64 |
| `V` | vocab size | 50257 |

> **主干张量形状永远是 `[B, T, C]`**。看到一个张量，先问"它是不是 `[B, T, C]`"。

**矩阵乘法唯一规则**：
$$
[a, b] \times [b, c] = [a, c]
$$
（内侧 `b` 必须相等，乘完消掉；外侧 `a, c` 保留。）

---

# 一、Attention — 全部公式

## 1.1 Single-Head Attention（单头）

### 核心公式（背下这一行）

$$
\boxed{\;\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V\;}
$$

### Q, K, V 怎么来的

输入 $X \in \mathbb{R}^{T \times C}$（一个句子，T 个 token，每个 token 是 C 维向量）。

通过 3 个**可学习矩阵**变出 Q, K, V：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

| 矩阵 | 形状 | 干啥 |
|---|---|---|
| $W_Q$ | `[C, C]` | "我要找什么" |
| $W_K$ | `[C, C]` | "我有什么" |
| $W_V$ | `[C, C]` | "我提供什么内容" |
| 输出 $Q, K, V$ | `[T, C]` | 三套不同的视角 |

### 形状追踪（逐步）

```
X       : [T, C]
W_Q     : [C, C]   →   Q = X W_Q : [T, C]
W_K     : [C, C]   →   K = X W_K : [T, C]
W_V     : [C, C]   →   V = X W_V : [T, C]

Kᵀ              : [C, T]                ← 转置
Q × Kᵀ          : [T, C] × [C, T] = [T, T]    ← 注意力分数
÷ √d_k          : [T, T]                ← 标量除法不改形状
softmax(dim=-1) : [T, T]                ← 每一行加起来 = 1
× V             : [T, T] × [T, C] = [T, C]    ← 加权求和

最终输出 : [T, C]   ← 和 X 同形状 ✅
```

> 🎯 注意力分数矩阵 $QK^\top$ 形状是 `[T, T]`，第 (i,j) 元素 = "token i 对 token j 的关注度"。

### 直觉：图书馆查书 📚

| 角色 | 类比 |
|---|---|
| **Query (q)** 🙋 | 你的问题："谁能告诉我我是啥？" |
| **Key (k)** 🏷️ | 每本书的标签 |
| **Value (v)** 📖 | 书的实际内容 |

**三步**：
1. `q · k` → 算每本书的相关度
2. `softmax` → 把相关度变成概率（加起来 = 1）
3. `× v` → 按相关度加权取出书的内容

### 各个部件的作用

| 部件 | 作用 |
|---|---|
| $QK^\top$ | 点积算相似度（i 对 j 的兴趣）|
| $/\sqrt{d_k}$ | 缩放，防 softmax 太尖、梯度消失 |
| softmax(dim=-1) | 沿"key 维"归一化为概率分布 |
| $\times V$ | 加权聚合 value |

### 为什么 Q ≠ K ≠ V

- $k$："**我有什么**"
- $q$："**我想找什么**"
- $v$："**具体给什么内容**"

**判断相关度**（用 k）和**取出内容**（用 v）是两件事，分开更灵活。

### Causal Mask（GPT/LLaMA 必备）

生成模型不能偷看未来 → 在 softmax **之前**把 $QK^\top$ 的**上三角**置为 $-\infty$：

```
       j=0   j=1   j=2   j=3        ← key 位置
i=0  [  ✓   -∞   -∞   -∞  ]        ← 只能看自己
i=1  [  ✓    ✓   -∞   -∞  ]
i=2  [  ✓    ✓    ✓   -∞  ]
i=3  [  ✓    ✓    ✓    ✓  ]
       ↑ query 位置
```

$-\infty$ 经 softmax → 0，那些位置的 V 就不会被取到。

---

## 1.2 Multi-Head Attention（多头）

### 核心公式

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

其中每个 head 独立计算：

$$
\text{head}_i = \text{Attention}(X W_Q^{(i)},\ X W_K^{(i)},\ X W_V^{(i)})
$$

### 为什么需要多头

**一组 Q/K/V 只能学一种"看的方式"**。多头让模型并行学多种关系：
- head 1 可能看语法
- head 2 可能看指代
- head 3 可能看远距离依赖
- ...

### 实现：切 C 维成 h 份

不真的做 h 次独立计算，而是把 `C = 768` 切成 `h = 12` 份，每份 `d_k = 64`：

```
X    : [T, 768]
   ↓ 一次性投影到 Q, K, V，每个还是 [T, 768]
   ↓ 但把 768 看成 12 × 64
   ↓ reshape 成 [T, h=12, d_k=64]
   ↓ transpose 成 [h=12, T, 64]   ← 把 head 维提前
   
[Q1, K1, V1] [Q2, K2, V2] ... [Q12, K12, V12]    每份 (T, 64)
   ↓ 各自独立做 attention（并行计算）
[O1] [O2] ... [O12]                              每份 (T, 64)
   ↓ 沿最后一维 concat 拼回去
O    : [T, 768]
   ↓ 再过一个 W_O (768×768) 做"混合"
最终输出 : [T, 768]
```

### 形状追踪（带 batch 和 head 维）

```
x      : [B, T, C]                  (B=2, T=5, C=768)
   │ W_Q, W_K, W_V (各是 [C, C])
   ▼
Q,K,V  : [B, T, C]
   │ 把 C 拆成 h × d_k → reshape
   ▼
Q,K,V  : [B, T, h, d_k]
   │ transpose(1, 2)
   ▼
Q,K,V  : [B, h, T, d_k]             ← h 提到第 2 位，方便并行
   │
   │ scores = Q @ K.transpose(-2,-1) / √d_k
   ▼
scores : [B, h, T, T]
   │ softmax(dim=-1)
   ▼
attn   : [B, h, T, T]
   │ attn @ V
   ▼
out    : [B, h, T, d_k]
   │ transpose(1,2) → reshape 把 h 和 d_k 合回 C
   ▼
out    : [B, T, C]
   │ × W_O ([C, C])
   ▼
最终    : [B, T, C]   ← 和输入同形状 ✅
```

### W_O 是什么、干什么

| 矩阵 | 形状 | 干啥 |
|---|---|---|
| $W_O$ | `[C, C]` | 把多头 concat 起来的输出"混合"一下 |

**为什么需要它**：concat 之后 12 个 head 是独立的，没有交互。$W_O$ 把这些独立信息线性组合一下，让 head 之间能"对话"。

### 参数量

一个 multi-head attention 模块的 weight：

| 矩阵 | 形状 | 参数量 |
|---|---|---|
| $W_Q$ | `[C, C]` | $C^2$ |
| $W_K$ | `[C, C]` | $C^2$ |
| $W_V$ | `[C, C]` | $C^2$ |
| $W_O$ | `[C, C]` | $C^2$ |
| **合计** | | **4 C²** |

$C = 768$ → 约 2.4M 参数/层。

### 多头 vs 单头：参数量一样吗？

**完全一样**。把单头的 $W_Q$（`[C, C]`）按列切成 12 块，每块是 `[C, d_k=64]` → 就是 12 个 head 的 $W_Q^{(i)}$。**没有额外参数，只是把同一个矩阵"逻辑切分"**。

---

## 1.3 PyTorch 代码骨架（背下来）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, C, h):
        super().__init__()
        assert C % h == 0
        self.h = h
        self.d_k = C // h
        # 一次性投影到 Q, K, V（合并成一个大矩阵更高效）
        self.W_qkv = nn.Linear(C, 3 * C, bias=False)
        self.W_O   = nn.Linear(C, C, bias=False)

    def forward(self, x):                       # x: [B, T, C]
        B, T, C = x.shape
        qkv = self.W_qkv(x)                     # [B, T, 3C]
        Q, K, V = qkv.chunk(3, dim=-1)          # 3 × [B, T, C]

        # 拆成多头
        Q = Q.view(B, T, self.h, self.d_k).transpose(1, 2)  # [B, h, T, d_k]
        K = K.view(B, T, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.h, self.d_k).transpose(1, 2)

        # 注意力分数 + 缩放 + softmax
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # [B, h, T, T]
        # （如果是 decoder：在这里加 causal mask）
        attn = scores.softmax(dim=-1)                            # [B, h, T, T]
        out  = attn @ V                                          # [B, h, T, d_k]

        # 拼回多头
        out = out.transpose(1, 2).contiguous().view(B, T, C)     # [B, T, C]
        return self.W_O(out)                                     # [B, T, C]
```

---

# 二、Normalization — 全部公式

## 2.1 LayerNorm

### 核心公式

$$
\boxed{\;\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\;}
$$

其中：

$$
\mu = \frac{1}{C} \sum_{c=1}^{C} x_c, \qquad
\sigma^2 = \frac{1}{C} \sum_{c=1}^{C} (x_c - \mu)^2
$$

> ⭐ **关键**：$\mu, \sigma$ **沿 C 维（最后一维）**计算，**每个 token 独立**做这套运算。

### 参数

| 符号 | 形状 | 是否可学 | 含义 |
|---|---|---|---|
| $\mu$ | `[B, T, 1]` | ❌ 算出来的 | 每个 token 自己的均值 |
| $\sigma^2$ | `[B, T, 1]` | ❌ 算出来的 | 每个 token 自己的方差 |
| $\gamma$ (weight) | `[C]` | ✅ 学习的 | 缩放（每个特征一个）|
| $\beta$ (bias) | `[C]` | ✅ 学习的 | 平移 |
| $\epsilon$ | 标量 | ❌ 固定 | 防除零（1e-5）|

### 5 步分解

```
输入 x : [B, T, C]
  ├─ 1. 沿最后一维算 μ  →  [B, T, 1]
  ├─ 2. 沿最后一维算 σ² →  [B, T, 1]
  ├─ 3. 减均值: x - μ  →  [B, T, C]
  ├─ 4. 除标准差: ... / √(σ² + ε)  →  [B, T, C]  ← 归一化后每个 token 均值=0, 方差=1
  └─ 5. 仿射: × γ + β  →  [B, T, C]              ← 给模型保留"调整空间"
输出 : [B, T, C]   ← 形状不变
```

### 为什么需要 γ, β（仿射变换）

如果只做归一化（步骤 1-4），**强制所有 token 都是均值 0 方差 1**，可能会损失表达力。
$\gamma, \beta$ 让模型**自己决定要不要这种归一化** — 极端情况下 $\gamma = \sigma, \beta = \mu$ 时输出 = 输入，完全"撤销"归一化。

### PyTorch 手写实现（今天对话讨论的）

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """Apply LayerNorm over the last dimension."""
    mean = x.mean(dim=-1, keepdim=True)                  # [B, T, 1]
    var  = x.var(dim=-1, keepdim=True, unbiased=False)   # [B, T, 1]
    x_norm = (x - mean) * torch.rsqrt(var + eps)          # [B, T, C]，rsqrt = 1/√x，更快
    return x_norm * gamma + beta                          # [B, T, C]
```

**5 个细节**：

| 点 | 选择 | 理由 |
|---|---|---|
| `dim=-1` | ✅ 必须沿最后一维 | LayerNorm 的灵魂 |
| `keepdim=True` | ✅ 必须 | 保留维度方便广播 `[B,T,1]` ↔ `[B,T,C]` |
| `unbiased=False` | ✅ 推荐 | 和 `nn.LayerNorm` 对齐（除以 N 不是 N-1）|
| `eps` 加在 var 内 | ✅ 正确 | $\sqrt{\sigma^2 + \epsilon}$ 数值稳定 |
| `rsqrt(...)` 替代 `1/sqrt(...)` | 💡 优化 | 更快、更稳，LLaMA 用这个 |

---

## 2.2 LayerNorm vs BatchNorm — 终极对比

### 公式上的区别（核心！）

**LayerNorm**（沿 C 维）：
$$
\mu_{(b,t)} = \frac{1}{C} \sum_{c=1}^{C} x_{b,t,c}
$$

**BatchNorm**（沿 B + T 维）：
$$
\mu_c = \frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} x_{b,t,c}
$$

| | LayerNorm | BatchNorm |
|---|---|---|
| **归一化的维度** | `C`（特征维）| `B + T`（跨样本 + 跨时间）|
| **μ, σ 的形状** | `[B, T, 1]` | `[1, 1, C]` |
| **算几组 μ, σ** | $B \times T$ 组（每个 token 一组）| $C$ 组（每个特征一组）|
| **依赖 batch 吗** | ❌ 不依赖 | ✅ 严重依赖 |
| **训练 / 推理一致** | ✅ 完全一致 | ❌ 推理用滑动平均 |
| **序列变长** | ✅ 完全 OK | ❌ 不同长度统计崩 |

### 几何可视化

把张量想成立方体 `[B, T, C]`：

```
         C (特征)
        ←─────────→
       ┌─────────┐
   B   │ ▓▓▓▓▓▓▓ │  ┐
   ↓   │ ▓▓▓▓▓▓▓ │  │  T (时间)
       │ ▓▓▓▓▓▓▓ │  ↓
       └─────────┘  

LayerNorm:                BatchNorm:
对每个 (b, t)：           对每个 c：
沿 C 方向算 μ, σ          沿 B+T 方向算 μ, σ
   ───►                      ▓
   ───►                      ▓ (一整列上下 + 一整页前后)
   ───►                      ▓
形状: B × T 组              形状: C 组
"token 自己跟自己比"        "跨样本跨时间比"
```

### 为什么 NLP / Transformer 必须用 LayerNorm

| 原因 | 说明 |
|---|---|
| ① 序列变长 | 5 词 vs 500 词，BatchNorm 沿 T 走会崩 |
| ② Batch 分布异质 | 问答+诗+代码混一起，BatchNorm 算 μ, σ 被污染 |
| ③ Padding 污染 | 短句补 0，BatchNorm 把 0 也算进 μ, σ |
| ④ 训推一致 | LayerNorm 无滑动平均，部署简单 |

### 经典对比例子（今天图里的）

```
Seq1: 1 2 3 2 3 1 2 2 3 1 2 3   →  batchnorm  →  正常被压扁
Seq2: 0 0 0 0 1 2 0 0 0 0 0 0   →  batchnorm  →  0.99 等夸张值 ❌
```

**原因**：Seq2 大部分是 0，让全局 μ, σ 都很小。归一化后 Seq2 那个突然的 `1, 2` 被**放大**成夸张值。
**LayerNorm 不会有这问题** — 每个 token 自己跟自己比，互不污染。

### CV vs NLP 的归一化选择

| 领域 | 用什么 | 为什么 |
|---|---|---|
| CV（ResNet, VGG）| BatchNorm | 图片尺寸固定、batch 分布相近 |
| NLP / Transformer | **LayerNorm** | 序列变长、分布异质 |
| 现代 LLM（LLaMA）| **RMSNorm** | LayerNorm 简化版，只算 σ 不算 μ |
| 小 batch | GroupNorm | BN 在小 batch 时不稳 |

---

## 2.3 Post-LN vs Pre-LN（位置之争）

**Post-LN**（原始 Transformer，2017）：
```
y = LayerNorm(x + Sublayer(x))
```

**Pre-LN**（现代 LLM 用的）：
```
y = x + Sublayer(LayerNorm(x))
```

| | Post-LN | Pre-LN |
|---|---|---|
| 训练稳定性 | 差，需要 warmup | **好，可以无 warmup** ⭐ |
| 性能上限 | 略高 | 略低（但差距很小）|
| 现代选择 | ❌ | ✅ GPT-2/3, LLaMA 全用这个 |

---

# 三、把所有公式放在一起 — 一页速查

## Attention 部分

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O
$$

$$
\text{head}_i = \text{Attention}(XW_Q^{(i)},\, XW_K^{(i)},\, XW_V^{(i)})
$$

## Normalization 部分

$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

$$
\mu = \frac{1}{C}\sum_c x_c, \quad \sigma^2 = \frac{1}{C}\sum_c (x_c - \mu)^2
$$

## 一个 Transformer Block（Pre-LN 版）

```
y = x + MultiHead(LayerNorm(x))
z = y + FFN(LayerNorm(y))
```

---

# 四、关键直觉总结（背下来）

1. ✅ **Attention = 软查字典**：Q 是问题，K 是标签，V 是内容；点积算相关度，softmax 归一化，加权取 V。
2. ✅ **Multi-Head = 并行多视角**：把 C 切成 h 份，每份独立 attention，最后拼回 + 过 W_O 混合。
3. ✅ **`dim=-1` 是 attention 和 softmax 的标配**：沿"最后一维"（key 维 / 特征维）做。
4. ✅ **LayerNorm 沿 C 维，每个 token 自己跟自己比** — 不依赖 batch、不被序列长度影响。
5. ✅ **BatchNorm 在 NLP 翻车**：因为序列变长、batch 分布异质、padding 污染。
6. ✅ **形状主干永远是 `[B, T, C]`**：attention 不改它，FFN 中间临时变 4C 再变回。

---

# 五、自检题（答得上才算懂）

1. **Q, K, V 形状是什么？$QK^\top$ 形状是什么？代表什么？**
   <details><summary>答</summary>
   都是 `[T, C]`（单头）或 `[B, h, T, d_k]`（多头）。$QK^\top$ 形状 `[T, T]`，第 (i,j) 元素 = token i 对 token j 的关注度。
   </details>

2. **为什么除以 $\sqrt{d_k}$？**
   <details><summary>答</summary>
   $d_k$ 大时点积值会很大 → softmax 输出非常尖锐（几乎是 one-hot）→ 梯度消失。除以 $\sqrt{d_k}$ 把方差拉回 1，softmax 输出平滑。
   </details>

3. **Multi-head 比 single-head 多了什么参数？**
   <details><summary>答</summary>
   **没多**（W_Q/W_K/W_V 总形状一样 `[C, C]`），只是把这些矩阵**逻辑切分**成 h 份让每份独立看。$W_O$ 是多头才有的，但单头也常加一个 output projection，所以总参数也差不多。
   </details>

4. **LayerNorm 沿哪一维做？为什么？**
   <details><summary>答</summary>
   沿最后一维 C（特征维 / model dim）。因为每个 token 独立归一化 → 不依赖 batch、不受序列长度影响、不被 padding 污染 → 完美适配 NLP 异质分布。
   </details>

5. **`x.var(dim=-1)` 默认 `unbiased=True`，标准 LayerNorm 要怎样？**
   <details><summary>答</summary>
   要 `unbiased=False`（除以 N 而不是 N-1）。和 `nn.LayerNorm` 对齐。差别小但要知道。
   </details>

6. **为什么 BatchNorm 在 NLP 翻车？**
   <details><summary>答</summary>
   (1) 序列变长，沿 T 维统计不稳；(2) batch 里句子分布异质，互相污染；(3) padding 的 0 污染统计量；(4) 训练/推理用不同统计量（推理用滑动平均），不一致。
   </details>

7. **Pre-LN 和 Post-LN 差别？**
   <details><summary>答</summary>
   Pre-LN 把 LayerNorm 放在 sublayer 之前（残差路径上没有 LN），训练稳定性好，无需 warmup，现代 LLM 全用这个。Post-LN 是原版 2017 的做法，训练比较敏感。
   </details>

---

# 六、扩展阅读

- [ ] **Karpathy nanoGPT** — 看 `model.py`，对照本文公式
- [ ] **The Illustrated Transformer**（Jay Alammar）— 最经典配图
- [ ] **RMSNorm 论文**（Zhang & Sennrich 2019）— 现代 LLM 标配
- [ ] **Flash Attention** — attention 的 GPU 优化版（CS336 lec 后期会讲）
- [ ] **On Layer Normalization in the Transformer Architecture**（Xiong et al. 2020）— Pre-LN 提出

---

> 🧭 **下一步建议**：
> 1. 在终端跑一遍 §1.3 的 PyTorch 代码，看形状打印对不对
> 2. 手写一遍 LayerNorm，和 `nn.LayerNorm` 对比输出
> 3. 看完回到 [`transformer-primer.md`](transformer-primer.md) §2-§6 互相印证
