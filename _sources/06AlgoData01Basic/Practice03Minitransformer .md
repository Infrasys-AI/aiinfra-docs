# 手把手实现迷你版 Transformer 
Author by: lwh

## Transformer 知识原理

Transformer 是一种基于 自注意力机制(Self-Attention)的深度学习模型结构，用于处理序列数据（如文本、语音、时间序列等）。相较于传统的循环神经网络(RNN、LSTM)，Transformer 有以下几个核心优势：

- 并行计算能力强：抛弃了序列依赖的结构，使训练更快；
- 长距离依赖建模能力强：自注意力机制可以直接建模任意位置间的关系；
- 结构模块化：由多层编码器和解码器堆叠构成，易于扩展和修改。

在本教程中，我们仅关注编码器（Encoder）部分的精简实现，其结构主要由以下模块组成：

1. 词嵌入（Embedding）：将输入的 token 序列（如词索引）映射为稠密向量，形成初始特征表示。
2. 位置编码（Positional Encoding）：由于 Transformer 完全抛弃了循环结构，因此需要手动注入位置信息。本实验使用论文提出的正余弦位置编码，使模型能够感知序列中 token 的位置信息。
3. 自注意力机制（Self-Attention）：通过计算每个位置之间的关系（注意力权重），让模型自主学习哪些位置更重要。其本质是对序列的不同部分进行加权组合，突出关键特征。
4. 残差连接 + 层归一化（Residual + LayerNorm）：为了解决深层网络中梯度消失和训练不稳定的问题，在注意力层和前馈网络后分别添加残差连接与 LayerNorm，增强模型表达能力。
5. 前馈网络（Feed-Forward Network）：对每个位置的表示分别通过两层全连接网络，进一步提取特征并引入非线性表达。

最终，编码器的每个模块都以“子层 → 残差连接 → 层归一化”的方式组成结构块，构成了一个可堆叠的 Transformer 编码器框架。


## Transformer 编码实现
首先我们导入所需的 PyTorch 和数学库：


```python
import torch
import torch.nn as nn
import math
```

与论文一致，我们使用固定的正余弦位置编码方式。注意编码维度必须为偶数。


```python
def sinusoidal_pos_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    if d_model % 2 != 0:
        raise ValueError("d_model 必须为偶数")

    # 生成位置向量和维度向量
    pos = torch.arange(0, seq_len).unsqueeze(1).float()        # shape: (seq_len, 1)
    i = torch.arange(0, d_model // 2).float()                  # shape: (d_model/2,)

    # 计算频率除数项
    denom = torch.pow(10000, 2 * i / d_model)                  # shape: (d_model/2,)
    angle = pos / denom                                        # shape: (seq_len, d_model/2)

    # 初始化编码矩阵
    pe = torch.zeros(seq_len, d_model)

    # 填入 sin 和 cos
    pe[:, 0::2] = torch.sin(angle)  # 偶数维度
    pe[:, 1::2] = torch.cos(angle)  # 奇数维度

    return pe
```

这个函数返回一个 `(seq_len, d_model)` 的位置编码张量，用于为输入序列添加位置信息。该位置编码函数完成后，我们就可以进入 Transformer 的核心：注意力机制的实现。点积注意力机制，它会使用三个输入张量：
 `q`（Query）：表示当前 token 想关注什么； `k`（Key）：表示序列中每个位置的“关键词”； `v`（Value）：表示每个位置实际携带的信息。
注意力机制会计算 `q` 与所有 `k` 的匹配程度，然后用这些权重对 `v` 进行加权平均，输出新的表示结果。



```python
def scaled_dot_product_attention(q, k, v):

    # 计算注意力打分矩阵：q 与 k 的转置点积，然后除以 sqrt(d_k) 进行缩放
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # shape: (..., seq_len_q, seq_len_k)

    # 对打分矩阵使用 softmax，得到注意力权重
    weights = torch.softmax(scores, dim=-1)  # shape: (..., seq_len_q, seq_len_k)

    # 使用注意力权重加权求和 v，得到注意力输出
    return weights @ v  # shape: (..., seq_len_q, d_v)

```

接下来我们构建一个迷你版的 Transformer 编码器类（MiniTransformerEncoder），它包含以下核心模块：
- 嵌入层（Embedding）：将输入的 token（整数索引）映射为 d_model 维的向量；
- 注意力层（Self-Attention）：使用缩放点积注意力机制提取全局上下文信息；
- 前馈网络（Feedforward）：对每个位置的表示进行非线性变换；
- 层归一化（LayerNorm）：分别应用在注意力子层和前馈子层之后，配合残差连接使用，有助于训练稳定。

此外，我们为每个位置添加了固定的 正余弦位置编码（Positional Encoding），使模型可以识别 token 的先后顺序。
注意：我们直接对词嵌入张量 `x_embed` 与位置编码 `pe` 使用加法（`x = x_embed + pe`），这是因为：
- `x_embed` 的形状是 `(1, seq_len, d_model)`（batch 维度为 1）；
- `pe` 的形状是 `(seq_len, d_model)`；
- 在 PyTorch 中，这两者可以自动广播（broadcasting）成相同形状，从而完成逐位置相加。




```python
class MiniTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.linear_q = nn.Linear(d_model, d_model)         # Q 映射层
        self.linear_k = nn.Linear(d_model, d_model)         # K 映射层
        self.linear_v = nn.Linear(d_model, d_model)         # V 映射层
        self.attn_output = nn.Linear(d_model, d_model)      # 注意力输出映射
        self.norm1 = nn.LayerNorm(d_model)                  # 第一个 LayerNorm

        self.ffn = nn.Sequential(                           # 前馈网络：两层全连接
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)                  # 第二个 LayerNorm

    def forward(self, x):
        seq_len = x.size(1)
        x_embed = self.embedding(x)                         # 获取词向量表示
        pe = sinusoidal_pos_encoding(seq_len, x_embed.size(-1))
        x = x_embed + pe                                    # 添加位置编码

        # 自注意力子层
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        attn = scaled_dot_product_attention(q, k, v)
        x = self.norm1(x + self.attn_output(attn))          # 残差连接 + LayerNorm

        # 前馈子层
        ff = self.ffn(x)
        x = self.norm2(x + ff)                              # 残差连接 + LayerNorm
        return x

```

这一结构就是一个基本的 Transformer 编码器块，具备捕捉上下文、感知位置、非线性变换的能力。
为了测试模型输出，我们用一个假输入 [3, 1, 7] 来运行前向传播：


```python
model = MiniTransformerEncoder(vocab_size=50, d_model=16)
dummy_input = torch.tensor([[3, 1, 7]])  # shape: [1, 3]
output = model(dummy_input)
print("=== 输出 ===")
print(output.detach().numpy())
print("=== 结束 ===")
```

    === 输出 ===
    [[[-1.5413152   1.6801366   0.05804453  0.24212095 -1.0774754
        0.02384599 -0.12485133 -0.6640581  -0.88564116  1.0022273
       -1.7447253  -0.11834528 -0.32912147  0.8888434   1.3979812
        1.1923339 ]
      [ 0.7319907   0.89123964  0.20003487 -0.07367807 -1.0843173
        0.2291747  -1.023271    0.09876721 -0.5554284   0.05615921
       -0.26637512 -1.0664971   0.01956824  1.8496377  -1.9267299
        1.9197246 ]
      [ 1.0861049  -0.38979262  1.8310152   0.47594324 -0.07559342
       -0.44681358 -1.166736    1.6186275  -1.5705074  -0.19102472
        0.40794823  1.0515163  -1.3678914  -0.19411024 -1.1033916
        0.03470579]]]
    === 结束 ===
    

至此，我们就完成了一个最小的 Transformer 编码器搭建。它结构清晰、功能完整，非常适合用作 Transformer 学习的入门代码框架。
