# Transformer 速成 — CS336 Lec 3 前置

> 🎯 **目标**：30 分钟内理解 vanilla Transformer 的全部组件，能看懂 LLaMA 一个 block 在做什么
> 📍 **定位**：CS336 Lec 3 的前置知识。Lec 3 默认你会本文所有内容。
> 🧭 **学习顺序**：① Token & Embedding → ② Self-Attention → ③ Multi-Head → ④ FFN → ⑤ LayerNorm & 残差 → ⑥ 位置编码 → ⑦ 组装

---

## 0. 一句话总览

> **Transformer = 把一串 token 反复"混合 + 加工"，最后预测下一个 token。**
> - 混合 = Self-Attention（每个 token 看其他 token）
> - 加工 = FFN（每个 token 自己过一个 MLP）
> - 反复 = 堆 N 层

---

## 1. Token & Embedding — 把文字变向量

### 流程

```
"the cat sat"
     ↓ Tokenizer
[464, 3797, 3332]              ← token ids（整数）
     ↓ Embedding 查表
[[0.1, -0.3, ..., 0.7],         ← 每个 token → d 维向量（如 d=768）
 [0.5,  0.2, ..., -0.1],
 [-0.4, 0.9, ..., 0.3]]
     ↓ shape: (seq_len=3, d=768)
```

**Embedding 是个查找表** `nn.Embedding(vocab_size, d)`：第 i 行就是 token id = i 的向量。这张表参数也会被训练。

### 形状约定（贯穿全文）

| 符号 | 含义 | 例子 |
|---|---|---|
| `B` | batch size | 32 |
| `T` | sequence length（token 数） | 1024 |
| `d` | hidden dimension | 768 |
| `h` | num heads | 12 |
| `d_k = d/h` | 每个 head 的维度 | 64 |
| `V` | vocab size | 50257 |

输入张量形状：**`(B, T, d)`**。后面所有操作都在这个形状上做。

---

## 2. Self-Attention — 让 token 互相看 ⭐

### 直觉

> 句子 "the cat sat on the mat"，理解 "sat" 时，模型应该多看 "cat"（动作的发出者），少看 "the"。**Self-attention 就是让每个 token 决定该多关注哪些其他 token，然后把它们的信息混进自己。**

### 三步走

**Step 1：从输入造 Q、K、V**

每个 token 向量 $x_i$ 经过三个**可学习矩阵** $W_Q, W_K, W_V$（都是 `d × d`）变出三个向量：

$$
q_i = x_i W_Q, \quad k_i = x_i W_K, \quad v_i = x_i W_V
$$

- $q_i$（Query）"我要找什么信息"
- $k_i$（Key）"我有什么信息"
- $v_i$（Value）"我具体提供什么内容"

整个序列一起做就是矩阵乘法：
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
形状全是 `(T, d)`。

**Step 2：算注意力权重**

$$
A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

- $QK^\top$ 形状 `(T, T)`：第 $i$ 行第 $j$ 列 = $q_i \cdot k_j$ = "token i 对 token j 的兴趣"
- $\div \sqrt{d_k}$：防止 softmax 太尖（数值稳定）
- `softmax(行)`：每一行归一化成概率分布（和 = 1）

**Step 3：加权求和**

$$\text{Output} = A V$$

每个 token 的新表示 = 所有 token 的 V 的加权平均，权重由 Q·K 决定。

### 完整公式（记住这一行）

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

### 类比

> **图书馆查书**：
> - 你的问题 = Query
> - 每本书的标签 = Key
> - 书的内容 = Value
> - 你和每本书的标签比一下相关度（Q·K），按相关度加权取出书的内容（×V）

---

## 3. Causal Mask — Decoder-Only 的关键

GPT / LLaMA 这种**生成模型**只能根据"前面已经写出的词"来预测下一个词，不能偷看未来。

实现：在 softmax **之前**，把 $QK^\top$ 矩阵的**上三角**置成 $-\infty$：

```
       j=0   j=1   j=2   j=3        ← key 位置
i=0  [  ✓   -∞   -∞   -∞  ]        ← 第 0 个 token 只能看自己
i=1  [  ✓    ✓   -∞   -∞  ]        ← 第 1 个 token 看 0, 1
i=2  [  ✓    ✓    ✓   -∞  ]
i=3  [  ✓    ✓    ✓    ✓  ]        ← query 位置
```

$-\infty$ 经过 softmax 变成 0 → 那些位置的 V 不会被取到。

---

## 4. Multi-Head Attention — 多个角度并行看

### 为什么不只用一个 head

一组 Q/K/V 只能学一种"看的方式"。多个 head 让模型并行学多种关系（如 head 1 看语法、head 2 看指代、head 3 看远距离依赖）。

### 怎么做

把 `d = 768` 切成 `h = 12` 份，每份 `d_k = 64`：

```
X (T, 768)
   ↓ 切成 12 份
[Q1, K1, V1] [Q2, K2, V2] ... [Q12, K12, V12]    每份形状 (T, 64)
   ↓ 各自独立做 attention
[O1] [O2] ... [O12]                              每份 (T, 64)
   ↓ 沿最后一维拼回去 (concat)
O (T, 768)
   ↓ 再过一个 W_O (768×768)
最终输出 (T, 768)
```

实现上 Q/K/V 投影矩阵直接用 `d × d`，再 reshape 成 `(T, h, d_k)`，并行计算。

---

## 5. FFN（前馈网络）— 每个 token 自己加工

Attention 之后，每个 token 独立过一个 **2 层 MLP**：

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

- $W_1$: `d × 4d`（升维到 4 倍）
- $W_2$: `4d × d`（降回原维度）
- 中间激活：ReLU（vanilla）/ GeLU / SwiGLU（现代）

**关键**：FFN **不混合 token**，每个位置独立计算。混合 token 的工作由 attention 包办，FFN 负责"加工每个 token 自己"。

> 💡 一个有用的视角：**Attention = 通讯，FFN = 思考**。

### 手算一遍 FFN（玩具尺寸 $d=3$，中间层 $=4$）

真实里 $d=768, 4d=3072$，机制完全一样，这里缩到能手算。

**输入**：某个 token 的向量

$$
x = \begin{bmatrix} 1.0 & -2.0 & 0.5 \end{bmatrix}
$$

**第一层**（升维 3 → 4）：

$$
W_1 = \begin{bmatrix}
1 & 0 & -1 & 2 \\
0 & 1 & 1 & 0 \\
-1 & 2 & 0 & 1
\end{bmatrix},\quad
b_1 = \begin{bmatrix} 0 & 0 & 0 & -1 \end{bmatrix}
$$

**第二层**（降维 4 → 3）：

$$
W_2 = \begin{bmatrix}
1 & 0 & 1 \\
-1 & 1 & 0 \\
0 & 1 & -1 \\
1 & -1 & 1
\end{bmatrix},\quad
b_2 = \begin{bmatrix} 0.5 & 0 & -0.5 \end{bmatrix}
$$

#### Step 1：$xW_1 + b_1$（线性升维）

$$
xW_1 = [\,0.5,\ -1.0,\ -3.0,\ 2.5\,]
\ \Rightarrow\ xW_1+b_1 = [\,0.5,\ -1.0,\ -3.0,\ 1.5\,]
$$

> 维度从 **3 升到 4**。纯线性。

#### Step 2：ReLU（逐元素砍负数）⭐

$$
\text{ReLU}\!\big([0.5,\ -1.0,\ -3.0,\ 1.5]\big) = [\,0.5,\ \mathbf{0},\ \mathbf{0},\ 1.5\,]
$$

| 位置 | 输入 | 输出 | 解读 |
|---|---|---|---|
| 0 | 0.5 | 0.5 | "探测器 0" 激活 ✅ |
| 1 | -1.0 | **0** | "探测器 1" 关闭 ❌ |
| 2 | -3.0 | **0** | "探测器 2" 关闭 ❌ |
| 3 | 1.5 | 1.5 | "探测器 3" 激活 ✅ |

> 🎯 **这一步是 FFN 的灵魂**：4 个"探测器"里只有 2 个亮灯。不同的 token 会点亮不同的子集 → 这就是非线性。

#### Step 3：$hW_2 + b_2$（线性降维回去）

$h = [0.5, 0, 0, 1.5]$：

$$
hW_2 = [\,2.0,\ -1.5,\ 2.0\,]
\ \Rightarrow\ \text{FFN}(x) = [\,2.5,\ -1.5,\ 1.5\,]
$$

#### 一图流

```
x = [1.0, -2.0, 0.5]                    ← 3维  (token 进来)
      │  × W₁ (3×4) + b₁
      ▼
   [0.5, -1.0, -3.0, 1.5]               ← 4维  线性升维
      │  ReLU 逐元素 ⭐
      ▼
   [0.5,  0.0,  0.0, 1.5]               ← 4维  ⚡负数被砍掉
      │  × W₂ (4×3) + b₂
      ▼
   [2.5, -1.5, 1.5]                     ← 3维  FFN(x) 输出
```

形状回到 3 维 → 可以接残差：`y = x + FFN(x)`。

#### 三个关键观察

1. **没有 ReLU 会塌掉**：$xW_1 W_2 + (b_1W_2+b_2)$ 直接合并成一个 $3\times3$ Linear，升维白做。**ReLU 是 FFN 唯一的非线性来源**。
2. **"4 倍升维"的意义**：中间层越宽 → ReLU 的"探测器候选"越多 → 能学到的特征组合越丰富。$d \to 4d$ 是经验最优。
3. **每个 token 独立**：输入是 `(T, d)` 时，FFN 对每一行独立做这套运算。**混合 token 的活儿是 attention 的，FFN 只管"每个 token 自己加工"**。

---

## 6. LayerNorm & 残差连接 — 训练能跑起来的关键

### LayerNorm

对每个 token 向量做"归一化"——让它的均值 = 0，方差 = 1，再用两个可学参数 $\gamma, \beta$ 缩放平移：

$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu, \sigma$ 沿**特征维 d** 计算（不是 batch 维）。

**作用**：保持每层输入分布稳定 → 深网络也能训。没有它，深 Transformer 训不起来。

### 残差连接（Residual）

每个子层都用 "输入 + 子层输出" 的形式：

```
y = x + Sublayer(x)         ← 而不是 y = Sublayer(x)
```

**作用**：
1. 梯度有"高速公路"直接回传 → 缓解梯度消失
2. 子层只需学**残差**（输出与输入的差），优化更容易

### 两种摆法（Lec 3 的重要话题）

**Post-LN**（vanilla，2017）：
```
y = LayerNorm(x + Sublayer(x))
```

**Pre-LN**（现代 LLM）：
```
y = x + Sublayer(LayerNorm(x))
```

差别看似很小，但 Pre-LN **训练稳得多**——这就是 Lec 3 会展开讲的内容。

---

## 7. 位置编码 — 让模型知道"谁在前谁在后"

Self-attention 本身**不感知顺序**："cat sat" 和 "sat cat" 算出来一样。所以要把位置信息**显式加进去**。

### Vanilla：Sinusoidal（正弦）位置编码

给每个位置 $p$ 一个固定的向量 $PE_p$（用 sin/cos 函数生成），直接加到 embedding 上：

$$x_p \leftarrow x_p + PE_p$$

### 现代：RoPE（Rotary Position Embedding）

不加到 embedding，而是在算 attention 时**旋转 Q 和 K**。

→ 细节是 **Lec 3 的内容**，现在不用懂。只要知道："位置编码就是告诉模型 token 的顺序，有好几种做法。"

---

## 8. 组装：一个完整的 Decoder-Only Transformer Block

```
输入 x  (B, T, d)
  │
  ├──→ LayerNorm ──→ Masked Multi-Head Self-Attention ──┐
  │                                                      │
  ◄──── (residual) ←─────────────────────────────────────┘
  │
  ├──→ LayerNorm ──→ FFN (Linear → activation → Linear) ─┐
  │                                                       │
  ◄──── (residual) ←──────────────────────────────────────┘
  │
输出 y  (B, T, d)
```

**这就是一个 block**。整个 Transformer = 堆 N 个这种 block（GPT-2 small: N=12，LLaMA-7B: N=32）。

### 完整流程（GPT 式 LM）

```
token ids        (B, T)
    ↓ Embedding + Position
hidden states    (B, T, d)
    ↓ N × Transformer Block
hidden states    (B, T, d)
    ↓ Final LayerNorm
    ↓ Linear to vocab_size (lm_head)
logits           (B, T, V)
    ↓ softmax
next-token 概率分布
```

训练时用 **cross-entropy loss**，对每个位置预测它的"下一个 token"。

---

## 🎯 自检 — 答得上来就可以开 Lec 3

1. **Q/K/V 是怎么来的？$Q K^\top$ 的形状是什么？代表什么？**
   <details><summary>答</summary>都是 $X$ 乘三个可学习矩阵得到。$QK^\top$ 形状 `(T, T)`，第 (i,j) 元素 = token i 的 query 和 token j 的 key 的点积 = "i 对 j 的关注度"。</details>

2. **为什么 decoder-only LM 需要 causal mask？怎么实现？**
   <details><summary>答</summary>训练时是并行预测每个位置的下一个 token，如果不 mask，模型会"偷看"未来的答案。实现：把 $QK^\top$ 上三角设为 $-\infty$，经 softmax 变 0。</details>

3. **一个 Transformer block 里 attention 和 FFN 的顺序？中间夹什么？**
   <details><summary>答</summary>顺序：Attention → FFN。每个子层都夹 LayerNorm 和残差连接（`y = x + Sublayer(LN(x))` 是 Pre-LN）。</details>

4. **Attention 和 FFN 各自负责什么？**
   <details><summary>答</summary>Attention 让 token 之间互相通讯、混合信息；FFN 让每个 token 独立"思考加工"（不混合）。</details>

5. **为什么需要位置编码？**
   <details><summary>答</summary>Self-attention 对 token 顺序不敏感（打乱顺序结果不变），位置编码显式注入位置信息。</details>

---

## 📚 推荐补充资源（可选）

- 🎬 **3blue1brown** "But what is a GPT?"（YouTube）— 15 分钟可视化
- 📖 **Jay Alammar** "The Illustrated Transformer"（博客）— 配图最好
- 💻 **Karpathy** "Let's build GPT from scratch"（YouTube 2h）— 完整实现，留到 Lec 3 学完再看

---

## 🧭 学完这份笔记 → 直接进 [`lec3-architectures.md`](lec3-architectures.md)

Lec 3 会讲"从 vanilla 到现代 LLM 的十几个改动"，本文是它的**对照基线**。
