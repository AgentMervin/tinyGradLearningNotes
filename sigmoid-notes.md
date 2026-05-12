# Sigmoid 函数学习笔记

## 1. 定义

Sigmoid（也叫 **logistic function**）：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- 输入：$z \in (-\infty, +\infty)$
- 输出：$\sigma(z) \in (0, 1)$
- 形状：S 形曲线，关于点 $(0, 0.5)$ 中心对称

| $z$ | $\sigma(z)$ | 含义 |
|---|---|---|
| $-\infty$ | $\to 0$ | 几乎肯定是负类 |
| $-2$ | $\approx 0.12$ | 较可能是负类 |
| $0$ | $0.5$ | 决策边界 |
| $+2$ | $\approx 0.88$ | 较可能是正类 |
| $+\infty$ | $\to 1$ | 几乎肯定是正类 |

---

## 2. 与逻辑回归的关系

逻辑回归并不是「直接用 Sigmoid 输出概率」，而是分两步：

1. **线性部分（logit）**：
$$
z = \mathbf{w}^\top \mathbf{x} + b \in (-\infty, +\infty)
$$
   这个 $z$ 称为 **logit / log-odds**，本身**不是概率**。

2. **Sigmoid 把 logit 映射成概率**：
$$
P(y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

### 为什么偏偏是 Sigmoid？

它来自 **对数几率 (log-odds)** 的反函数。逻辑回归的核心假设是：

$$
\underbrace{\log \frac{p}{1 - p}}_{\text{log-odds}} = \mathbf{w}^\top \mathbf{x} + b
$$

对两边解出 $p$，就自然得到：

$$
p = \frac{1}{1 + e^{-z}} = \sigma(z)
$$

所以 Sigmoid 不是「随便拍脑袋选的压缩函数」，而是**「假设 log-odds 是线性的」这一建模选择的必然结果**。

---

## 3. 与 Softmax 的关系

**Softmax 是 Sigmoid 在多分类下的推广**；反过来说，**Sigmoid 是 Softmax 在二分类下的特例**。

二分类 Softmax：
$$
P(y=1) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)
$$

| 任务 | 激活函数 | 输出维度 | 损失函数 |
|---|---|---|---|
| 二分类 | Sigmoid | 1 | Binary Cross-Entropy (BCE) |
| 多分类（互斥） | Softmax | $K$ | Categorical Cross-Entropy |
| 多标签（不互斥） | 每维独立 Sigmoid | $K$ | 每维独立 BCE |

⚠️ **多标签分类（一个样本可属多个类）要用 Sigmoid，不要用 Softmax**，因为 Softmax 强制 $\sum_k p_k = 1$，会让类别之间「抢概率」。

---

## 4. 导数（反向传播必备）

$$
\sigma'(z) = \sigma(z) \bigl(1 - \sigma(z)\bigr)
$$

**推导**：
$$
\sigma(z) = (1 + e^{-z})^{-1}
$$
$$
\sigma'(z) = -(1+e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}
$$
注意到 $\frac{e^{-z}}{1+e^{-z}} = 1 - \sigma(z)$，所以：
$$
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)
$$

**反向传播中的好处**：前向算出的 $a = \sigma(z)$ 可以直接复用，无需重新计算指数：
```python
a = sigmoid(z)            # forward
grad = a * (1 - a) * dL_da  # backward
```

### 导数的最大值

- $\sigma'(0) = 0.25$（在 $z=0$ 处取最大）
- $|z|$ 越大，$\sigma'(z)$ 越接近 0 → **梯度消失**

---

## 5. 配 BCE 损失：梯度变得超干净

二分类交叉熵：
$$
L = -\bigl[y \log \hat{y} + (1-y)\log(1-\hat{y})\bigr], \quad \hat{y} = \sigma(z)
$$

对 $z$ 求导（链式法则展开后大量项消掉）：
$$
\boxed{\frac{\partial L}{\partial z} = \hat{y} - y}
$$

> 这就是为什么 Sigmoid + BCE 是「天作之合」：梯度形式极其简洁，且**不会**出现 Sigmoid 自身导数 $\sigma(z)(1-\sigma(z))$ 带来的额外衰减项。

如果错误地用 Sigmoid + MSE，梯度会变成：
$$
\frac{\partial L}{\partial z} = (\hat{y} - y) \cdot \sigma'(z)
$$
当预测严重错误时（比如 $y=1$ 但 $\hat{y} \approx 0$），$\sigma'(z) \approx 0$，梯度几乎消失，**学不动** —— 这是分类任务不该用 MSE 的核心原因之一。

---

## 6. 常见陷阱

### 6.1 梯度消失
深层网络中堆叠 Sigmoid 会让梯度在反向传播时被反复乘上 $\le 0.25$ 的因子，迅速衰减到 0。
- 现代隐藏层普遍改用 **ReLU / GELU / SiLU**
- Sigmoid 主要保留在**输出层**（二分类、门控如 LSTM/GRU）

### 6.2 输出非零中心
$\sigma(z) \in (0, 1)$，恒为正 → 下一层输入恒为正 → 权重梯度方向受限（同号），收敛变慢。
- Tanh 是「零中心版」的 Sigmoid：$\tanh(z) = 2\sigma(2z) - 1$

### 6.3 数值稳定性
- $z$ 很大正数时：$e^{-z}$ 下溢 → 还好，结果约等于 1
- $z$ 很大负数时：直接算 $\frac{1}{1+e^{-z}}$ 会让 $e^{-z}$ 上溢 → 用等价形式：

```python
def sigmoid_stable(z):
    # 正负分别处理，避免 exp 溢出
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )
```

实际工程里更常用 **`log-sigmoid`** 或框架自带的 `binary_cross_entropy_with_logits`，它们把 log 和 sigmoid 融合，避免分别计算的数值问题。

---

## 7. 一句话总结

> **Sigmoid = 把任意实数 logit 翻译成 (0,1) 概率的「log-odds 反函数」。**
> 它是逻辑回归的输出层、二分类 Softmax 的特例、LSTM/GRU 的门控单元；配 BCE 用，别配 MSE 用；深层隐藏层别用，会梯度消失。

---

## 参考阅读

- `softmax-notes.md`（多分类视角）
- 《Deep Learning》(Goodfellow et al.) Ch. 6.2 — 输出单元
- CS231n: Neural Networks Part 1 — Activation Functions
