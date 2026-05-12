# Softmax 入门笔记

> 这份笔记尽量用「人话 + 小例子」讲清楚 softmax，公式都配中文解释。
> 在 VS Code 里按 `Cmd+Shift+V` 打开 Markdown 预览，公式会被渲染成漂亮的数学符号。

---

## 1. Softmax 是什么？

一句话：**把一组任意大小的数字，变成一组「加起来等于 1」的概率。**

常用在分类网络的最后一层。比如要识别一张图是猫/狗/鸟，网络最后会吐出 3 个数字（叫 logits），softmax 把它们变成 3 个概率。

### 公式

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

逐项翻译：

- $z = (z_1, z_2, \dots, z_n)$：网络输出的一组数字（logits）
- $e^{z_i}$：把每个数字取指数（让它变正数，并且大的更大、小的更小）
- 分母 $\sum_j e^{z_j}$：所有指数加起来，做归一化
- 结果 $\text{softmax}(z_i)$：第 $i$ 类的概率

### 小例子

假设网络输出 $z = (2.0,\ 1.0,\ 0.1)$（猫/狗/鸟的得分）。

| 步骤 | 猫 | 狗 | 鸟 |
|---|---|---|---|
| 原始 logits $z_i$ | 2.0 | 1.0 | 0.1 |
| 取指数 $e^{z_i}$ | 7.39 | 2.72 | 1.11 |
| 除以总和 11.22 | **0.659** | **0.242** | **0.099** |

三个概率加起来 = 1 ✅，模型最相信是猫（66%）。

### 数值稳定版（实际代码都这么写）

直接算 $e^{z_i}$ 容易溢出（比如 $z_i = 1000$）。技巧：先减去最大值，结果完全一样：

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

---

## 2. Forward 和 Backward 是什么？

> ⚠️ 你说的 "backend" 应该是 **backward**（反向传播）。
> "backend" 指的是框架底层（CPU/GPU 执行引擎），是另一个概念。

神经网络训练就是不停重复这两步：

### Forward（前向传播）= 算预测

数据从输入 → 一层层往后算 → 得到预测结果 → 算出 loss（预测和真实答案的差距）。

```
输入图片 → [卷积] → [全连接] → logits → [softmax] → 概率 → [交叉熵] → loss
                          forward 方向 →→→
```

### Backward（反向传播）= 算梯度

从 loss 开始，**反着走**，用链式法则算出「每个参数应该往哪个方向调」，也就是梯度 $\dfrac{\partial L}{\partial \theta}$。然后优化器（SGD、Adam）拿这些梯度去更新参数。

```
loss → [交叉熵的梯度] → [softmax的梯度] → [全连接的梯度] → [卷积的梯度]
                          ←←← backward 方向
```

### 类比

- **Forward** 像考试：拿到题目，一步步算出答案，对照标准答案得到「错了多少分」。
- **Backward** 像复盘：从「错了多少分」反推「哪一步算错了，下次该怎么改」。

---

## 3. 什么时候要求偏导数？

**只在 backward 阶段求偏导。** Forward 阶段只算数值，不碰梯度。

### 为什么要求偏导？

因为我们要做梯度下降：

$$
\theta_{\text{新}} = \theta_{\text{旧}} - \eta \cdot \frac{\partial L}{\partial \theta}
$$

- $\theta$：参数（权重）
- $\eta$：学习率
- $\dfrac{\partial L}{\partial \theta}$：loss 对参数的偏导，告诉我们「调大还是调小、调多少」

没有偏导，就不知道往哪边调，模型就学不会。

### Softmax 自己的偏导（看不懂可以跳过）

输出 $p_i$ 对输入 $z_j$ 的偏导分两种情况：

- 当 $i = j$（对自己求导）：$\dfrac{\partial p_i}{\partial z_i} = p_i (1 - p_i)$
- 当 $i \neq j$（对别人求导）：$\dfrac{\partial p_i}{\partial z_j} = -\, p_i\, p_j$

### 真实情况：Softmax + 交叉熵 一起算，超级简洁 ✨

实际中 softmax 后面几乎总是接交叉熵 loss。两者合并求导后，结果非常漂亮：

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

翻译成人话：

- $p_i$：softmax 算出的「模型认为是第 i 类的概率」
- $y_i$：真实标签（one-hot，正确类是 1，其它是 0）
- **梯度 = 预测概率 − 真实标签**

继续上面猫狗鸟的例子，假设真实答案是「猫」，即 $y = (1, 0, 0)$：

| | 猫 | 狗 | 鸟 |
|---|---|---|---|
| 预测 $p_i$ | 0.659 | 0.242 | 0.099 |
| 真实 $y_i$ | 1 | 0 | 0 |
| 梯度 $p_i - y_i$ | **−0.341** | **+0.242** | **+0.099** |

直觉解释：
- 猫的 logit 要**调大**（梯度为负，下降时减去负数 = 加）
- 狗、鸟的 logit 要**调小**

这就是为什么 PyTorch 的 `nn.CrossEntropyLoss`、TensorFlow 的 `softmax_cross_entropy_with_logits` 都把 softmax 和交叉熵合并实现 —— **又快又稳定**。

---

## 4. 在 tinygrad 里怎么对应

```python
import tinygrad as tg

# === Forward ===
logits = model(x)          # 网络前向，得到 logits
loss   = logits.sparse_categorical_crossentropy(y)  # 内部含 softmax

# === Backward ===
loss.backward()            # 自动反向传播，算出所有参数的梯度

# === 更新 ===
optimizer.step()           # 用梯度更新参数
```

- `forward`：调用 `model(x)`、`loss = ...` 时构建计算图、算出数值。
- `backward`：调用 `loss.backward()` 时，tinygrad 沿计算图反向，对每个算子套用它的梯度公式。
- **偏导计算时机**：仅在 `.backward()` 被触发时。

---

## 一句话总结

> **Softmax** 把一组数字变成概率；
> **Forward** 算预测和 loss，**Backward** 求偏导更新参数；
> 偏导**只在 backward 阶段算**，softmax + 交叉熵合在一起的梯度就是简洁的 **`预测 − 真实`**。
