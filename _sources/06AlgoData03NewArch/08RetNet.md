<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# RetNet 架构、留存机制与多计算范式深度解析：循环、并行、分块及硬件融合

> Author by: 张嘉瑶

## RetNet 的架构论点：破解序列建模的“不可能三角”


大型语言模型（LLM）领域长期被一个核心的架构“不可能三角”（The Impossible Trilemma）所困扰。这个三难困境（Trilemma）指出，一个理想的序列模型架构难以同时实现三个关键特性：



1. 高效的训练并行性 (Training Parallelism)：充分利用现代硬件（如 GPU）的大规模并行计算能力进行快速训练。
2. 低成本的推理 (Low-Cost Inference)：在生成序列时，每个步骤的计算和内存复杂度应尽可能低，理想情况下为 $O(1)$。
3. 强大的模型性能 (Strong Performance)：在各种任务和基准上，至少能媲美甚至超越现有SOTA（State-of-the-Art）架构（即 Transformer）的性能和扩展性。

现有的主流架构往往只能满足其中的两项：

Transformer：通过其自注意力机制（Self-Attention），实现了卓越的训练并行性（1）和强大的性能（3）。但其推理效率低下：由于需要维护一个不断增长的键值缓存（Key-Value Cache），其每一步推理的计算复杂度为 $O(N)$，内存复杂度为 $O(N^2)$ 或 $O(N)$（取决于缓存策略），导致部署成本高昂。

循环神经网络 (RNN / LSTM)：通过其循环状态（Recurrent State），实现了 $O(1)$ 的低成本推理（2）。但其固有的顺序依赖性使其难以在训练期间进行并行化（1），并且受限于梯度消失等问题，其长距离依赖建模能力和整体性能（3）远逊于 Transformer。

Retentive Network（RetNet） 正是为解决这一根本矛盾而提出的。它被设计为 Transformer 的潜在“继任者” ，其核心论点是：通过一种新的“留存机制”（Retention Mechanism），可以在一个统一的框架内同时实现上述所有三个目标。


<div align="center">
  <img src="../images/06AlgoData03NewArch/retnet2.jpg" alt="img" />
  <figcaption><strong>RetNet实现了“不可能三角”，它同时实现了训练并行性、良好性能和低推理成本。</strong></figcaption>
</div>



从架构上看，RetNet 延续了 Transformer 的宏观设计，由 $L$ 个相同的块（Block）堆叠而成 11。每个块包含两个核心子层（Sub-layer）：
1. 多尺度留存 (Multi-Scale Retention, MSR)：这是 RetNet 的核心创新，用以替代 Transformer 的多头注意力（Multi-Head Attention）。
2. 前馈网络 (Feed-Forward Network, FFN)：通常采用标准的 FFN 结构（如 SwiGLU），与 Transformer 中的 FFN 模块类似。

与 Transformer 一样，RetNet 在每个子层周围都应用了层归一化（Layer Normalization）和残差连接（Residual Connections）。这种设计的背后，隐藏着对行业需求的深刻洞察。三难困境不仅是一个技术挑战，更是一个严峻的经济挑战。随着模型规模和应用需求的增长，AI 推理（Inference）的成本已成为企业部署 LLM 时的主要障碍和“被忽视的成本挑战”。Transformer 的 $O(N)$ 推理复杂性导致其运营支出（OpEx）居高不下。RetNet 声称通过 $O(1)$ 的推理复杂度直接解决了这一经济痛点，旨在从根本上降低 AI 服务的总拥有成本（TCO）。

<div align="center">
  <img src="../images/06AlgoData03NewArch/retnet1.jpg" alt="img" />
  <figcaption><strong>RetNet与Transformer相比，实现了低成本推理（即GPU内存、吞吐量和延迟方面）、训练并行性以及良好的缩放曲线。推理成本的结果是以8k作为输入长度来报告的。</strong></figcaption>
</div>



## 留存机制 (Retention Mechanism) 的数学基础

RetNet 的精妙之处在于其核心机制——留存（Retention）——的数学构造。与“并行优先”的 Transformer 不同，RetNet 的并行形式是从一个循环（Recurrent）形式推导出来的，从而在两者之间建立了数学上的等价性 。   

**理论推导：从循环到并行的对偶性**

留存机制的推导始于一个标准的 RNN 状态转移方程 ：

$$\mathbf{s}_n = A \mathbf{s}_{n-1} + K^\intercal_n \mathbf{v}_n$$

$$\mathbf{o}_n = Q_n \mathbf{s}_n = \sum_{m=1}^{n} Q_n A^{n-m} K^\intercal_m \mathbf{v}_m$$

其中，$\mathbf{s}_n$ 是 $n$ 时刻的循环状态，$\mathbf{o}_n$ 是输出。$Q_n, K_n, \mathbf{v}_n$ 是输入 $X_n$ 经过不同投影（Projection）得到的值。关键的转变发生在对状态转移矩阵 $A$ 的处理上。通过引入一个对角化的、旋转的状态矩阵 $A = \Lambda(\gamma e^{i\theta}) \Lambda^{-1}$，并将其简化为标量衰减因子 $\gamma$ 和旋转项 $e^{i\theta}$，上述求和公式可以被重写为:

> 精彩亮点： RetNet用位置矩阵替代了原始Transformer中的softmax。

$$\mathbf{o}_n = \sum_{m=1}^{n} \gamma^{n-m} (Q_n e^{in\theta})(K_m e^{-im\theta})^\dagger \mathbf{v}_m$$


这个基于求和的公式可以进一步被重构为一个并行的矩阵运算形式，即留存机制的并行表示 (Parallel Representation) ：

$$\text{Retention}(X) = (Q K^\intercal \odot D) V$$

其中：
* $Q = (X W_Q) \odot \bar{\Theta}$
* $K = (X W_K) \odot \Theta$
* $V = X W_V$
* $\Theta_n = e^{in\theta}$
* $D$ 是一个结合了因果掩码（Causal Masking）和指数衰减的矩阵，其元素 $D_{nm} = \gamma^{n-m}$ (当 $n \ge m$ 时) 或 0 (当 $n < m$ 时)。


**核心组件 1：指数衰减 ($\gamma$) 与多尺度留存 (MSR)**

$\gamma$（Gamma）是留存机制的灵魂。它是一个固定的指数衰减超参数，取代了 Transformer 中依赖数据的、计算昂贵的 $\text{softmax}$ 函数。

“多尺度”（Multi-Scale）的含义是，在 MSR 模块中，每个“头”（Head）被分配了一个不同且固定的 $\gamma$ 值。这种设计源于一个深刻的洞察：与标准注意力机制（Attention）让模型学习关注不同距离的上下文不同，RetNet 通过 $\gamma$ 植入了一种硬编码的归纳偏置（Inductive Bias）。

模型被架构性地强制分配不同的记忆尺度：
* $\gamma$ 值接近 1 的头（慢衰减），被迫专注于长程依赖（长期记忆）。
* $\gamma$ 值较小的头（快衰减），被迫专注于局部上下文（短期记忆）。

这种设计降低了模型的优化负担，但也可能牺牲了一定的灵活性。在实践中，$\gamma$ 的值按如下公式设定：

$$\gamma = 1 - 2^{-5-\text{arange}(0, h)} \in R^h$$

其中 $h$ 是头的数量。

**核心组件 2：位置编码 ($\Theta$) 与门控机制 (Gating)**

位置编码：在并行表示中，$\Theta_n = e^{in\theta}$ 项被逐元素地应用于 $Q$ 和 $K$ 矩阵。这本质上是一种旋转位置编码（Rotary Position Embedding, RoPE）的变体。RetNet 的论文也指出，这种推导与 xPos（一种相对位置编码）的思想是一致的。

门控机制：MSR 模块的输出 $Y$（经过 GroupNorm）会通过一个门控机制来增强非线性表达能力。该机制由一个 $\text{swish}$ 激活函数和一个额外的线性投影 $W_G$ 控制：

$$\text{MSR}(X) = (\text{swish}(X W_G) \odot Y) W_O$$

消融实验（Ablation studies）证实，$\text{swish}$ 门控和 GroupNorm 对于 RetNet 取得高性能至关重要。


<div align="center">
  <img src="../images/06AlgoData03NewArch/retnet4.jpg" alt="img" />
  <figcaption><strong>消融实验</strong></figcaption>
</div>



**机制对比：Retention vs. Self-Attention**

RetNet 的设计巧妙地融合了位置和衰减。在 Transformer 中，位置编码（如 RoPE）和注意力得分（$\text{softmax}$）是两个独立计算的组件。而在 RetNet 中，留存机制本身就是一种位置感知的衰减机制。它内置了两种相对位置偏置：
1. 旋转偏置 ($\Theta$)：通过绝对位置 $n$ 的旋转来编码相对位置信息。
2. 衰减偏置 ($D$)：通过相对距离 $n-m$ 来显式地施加指数衰减。

$$Transformer Attention :\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_{k}}}\right)V$$
 $$RetNet Retention (Parallel): \text{Retention}(X) = (Q K^\intercal \odot D) V$$


<div align="center">
  <img src="../images/06AlgoData03NewArch/retnet5.jpg" alt="img" />
  <figcaption><strong>左侧为 Transformer 计算，右侧为 RetNet 计算。</strong></figcaption>
</div>



RetNet 的核心是用一个逐元素的 Hadamard 乘积 $(\odot D)$ 替换了 $\text{softmax}(\cdot)$ 及其缩放因子。
这一改动带来了深远的影响：
* 数学对偶性：$\text{softmax}$ 是一个非线性操作，它破坏了计算的序列可分解性。通过移除 $\text{softmax}$，RetNet 实现了并行形式和循环形式之间的严格数学等价。
* 性能考量：尽管有观点担心“一切都线性化”可能会损害模型的表达能力，但 RetNet 的实证结果表明，MSR、门控机制和 FFN 层的结合，足以补偿 $\text{softmax}$ 的缺失，并在 2B 以上参数规模的模型上开始超越 Transformer。 

## RetNet 的三种计算范式 (Multi-Computation Paradigms)
RetNet 架构的基石在于其能够在三个不同的计算范式之间自由切换，而共享同一套模型权重 。这三种范式分别针对训练、推理和长序列训练进行了优化，完美地解决了“不可能三角”的三个角。


<div align="center">
  <img src="../images/06AlgoData03NewArch/retnet6.jpg" alt="img" />
  <figcaption><strong>留存率三种计算范式的伪代码</strong></figcaption>
</div>


**范式 1：并行表示 (Parallel Representation)** 
* 公式 4: $\text{Retention}(X) = (Q K^\intercal \odot D) V$
* 目的：高效训练。
* 分析：此范式专为 GPU 并行计算设计。它将所有计算转换为大规模矩阵乘法（MatMul）和逐元素操作（Element-wise），这是现代深度学习框架中优化最充分的操作。尽管 $D$ 矩阵的计算和存储（$O(N^2)$）使其在理论上仍是二次复杂度，但它省去了 $\text{softmax}$ 的复杂计算和 I/O 开销，因此在实际训练中依然高效。
* 解决：三难困境中的“训练并行性”。

**范式 2：循环表示 (Recurrent Representation)**

* 公式: 对于第 $n$ 个时间步：
$$\mathbf{S}_n = \gamma \mathbf{S}_{n-1} + K_n^\intercal V_n $$ $$ \text{Retention}(X_n) = Q_n \mathbf{S}_n$$
* 目的：低成本推理。
* 分析：这是 RetNet 的“杀手锏”特性。在自回归生成（Autoregressive Inference）任务中，模型只需计算当前时间步 $n$。
    1. 状态 $\mathbf{S}_n$ 是一个固定大小的矩阵（其维度仅与 $d_{\text{model}}$ 相关，与序列长度 $N$ 无关）。
    2. 每一步的计算仅涉及一次小规模的矩阵-向量乘法和一次加法。
* 复杂度：$O(1)$ 的计算和内存复杂度。
* 对比：Transformer 的推理是 $O(N)$，因为注意力必须作用于全部的 KV 缓存。这意味着 Transformer 的推理延迟随序列长度线性增长，而 RetNet 的推理延迟是与序列长度无关的常数。
* 解决：三难困境中的“低推理成本”。

<div align="center">
  <img src="../images/06AlgoData03NewArch/retnet3.jpg" alt="img" />
  <figcaption><strong>RetNet 用于推理的循环计算</strong></figcaption>
</div>


RetNet 最根本的创新在于，范式 1（并行）和范式 2（循环）在数学上是完全等价的 。它们是计算同一个函数的两种不同方式。这使得“并行训练，循环部署” (Train-Parallel, Deploy-Recurrent) 策略成为可能 。模型可以在 GPU 集群上利用并行范式进行高效训练，然后（无需任何修改或重新训练）直接转换为超高效的循环范式，用于边缘设备或服务器的低成本部署。这在理论上完美地解决了 Transformer/RNN 的核心悖论。

**范式 3：分块循环表示 (Chunkwise Recurrent Representation)**

* 机制：一种混合（Hybrid）计算模式，旨在平衡并行计算速度和内存消耗。
1. 序列被划分为固定长度 $B$ 的“块”（Chunks）。
2. 块内计算 (Intra-chunk)：在每个块内部，使用并行表示（范式 1）进行快速计算
3. 块间计算 (Inter-chunk)：在块与块之间，使用循环表示（范式 2）来传递一个压缩的循环状态 $R_i$。
* 目的：高效的长序列训练。
* 分析：这是针对训练阶段的工程优化。范式 1 的 $O(N^2)$ 内存对于极长的序列（例如 $N > 16k$）是不可接受的。范式 2 的 $O(N)$ 顺序计算对于训练来说又太慢。分块循环范式通过 $O(N \cdot B)$ 的计算复杂度，在块大小 $B$（例如 512）为常数的情况下，实现了线性 $O(N)$ 的训练复杂度。
* 解决：在保持并行训练优势的同时，实现对超长序列的高效建模



## 硬件加速与核函数融合 (Kernel Fusion) 分析

对 RetNet 的性能分析必须始于一个关键事实：原始 RetNet 论文中的惊人基准测试（Benchmarks）是使用“vanilla PyTorch code”实现的。作者明确指出，诸如“核函数融合或类 FlashAttention 的加速”是“未来的工作”。

尽管如此，RetNet（无融合）依然取得了：
* 推理：在 7B 模型和 8k 序列长度下，比标准 Transformer 快 8.4 倍，节省 70% 显存。
* 训练：即使与高度优化的 FlashAttention v1 相比，RetNet 仍在训练速度和显存占用上显示出“优势”。

这一发现至关重要。它表明 RetNet 的性能提升首先来源于其算法设计（移除 $\text{softmax}$，启用 $O(1)$ 推理）。这种算法增益是独立于、且可以叠加于硬件级优化的。在应用任何底层优化技巧之前，RetNet 架构本身就比 Transformer 更高效。

要理解 RetNet 如何“融合”，必须先理解 FlashAttention 2  的优化原理。Attention 和 Retention 这类操作都是内存带宽受限 (Memory-Bound) 的，而非计算受限 (Compute-Bound) 。计算单元（ALU）的等待时间远远超过其计算时间，瓶颈在于数据在不同层级显存（HBM 和 SRAM）之间的读写（I/O）。

FlashAttention 2 的核心优化包括：
* 核函数融合 (Kernel Fusion)：将多个操作（如 MatMul $\rightarrow$ Softmax $\rightarrow$ Dropout $\rightarrow$ MatMul）合并到一个 GPU 核函数中，大幅减少对 HBM（高带宽显存）的读写次数。
* Tiling 与重计算：将 $N \times N$ 的注意力矩阵切片（Tiling）为适合 GPU 内部高速缓存（SRAM）的小块，并在 SRAM 中进行计算，通过重计算（Recomputation）来避免将这个巨大的中间矩阵写回 HBM。

“RetNet 融合 FlashAttention 2” 是一个概念性的目标，而非字面上的实现。RetNet 不能 直接使用 FlashAttention 2 的代码库。像 FlashAttention 这样的库具有“严格的约束”并“绑定到固定配置”。RetNet 的算法（无 $\text{softmax}$，但有 $\odot D$）和“非典型输入维度”使其与 FlashAttention 2 不兼容。

因此，业界采用了更先进的策略来实现 RetNet 的硬件加速：
* 方案 1：专用融合库 (如 fla-org/flash-linear-attention)像 fla 这样的库 应运而生，它们专为线性注意力变体（包括 RetNet, Mamba, RWKV 等）提供硬件加速。这些库使用 Triton（一种比 CUDA 更灵活的 GPU 编程语言）来编写新的、定制化的融合核函数。它们将 FlashAttention 的I/O 感知原理应用到了 RetNet 的特定算法上。
* 方案 2：通用融合框架 (如 AttentionEngine)最新的研究认识到，为每一种新架构（RetNet, Mamba 等）手写优化核函数是不可持续的。AttentionEngine 提出了一个解决方案：一个抽象的框架，它将所有注意力机制（包括 Retention）分解为两个基本操作：“相关性评分”（Relevance Scoring）和“聚合”（Aggregation）。

该框架随后可以自动为任何新机制（包括 RetNet 的并行、循环和分块范式）生成优化的、融合的核函数，并能妥善处理其“非典型维度” 。

综上所述，受益于 FlashAttention 2 所开创的 I/O 感知融合原理。RetNet（以及 Mamba）的出现，正在推动 AI 硬件生态系统从手写的、单一的核函数（如 FlashAttention 2），转向更通用的、基于编译器的优化框架（如 AttentionEngine），以适应后 Transformer 时代架构的多样性。

## 架构的横向对比分析
RetNet 的价值需要通过与 Transformer 和其他竞争者（如 Mamba）的直接对比来体现。

**RetNet vs. Transformer (定量性能)**

原始论文提供了详尽的定量对比，证明了 RetNet 在解决三难困境方面的有效性。以下表格总结了其与标准 Transformer 及 FlashAttention v1 优化版 Transformer 的性能对比。


<div align="center">
  <img src="../images/06AlgoData03NewArch/retnet7.jpg" alt="img" />
</div>



RetNet vs. Mamba (机制深度对比)

Mamba 是 RetNet 之外最受瞩目的 Transformer 挑战者。两者都利用了 RNN 的思想，但实现路径截然不同。

Mamba (选择性状态空间模型 - SSSM):
* 机制：Mamba 的核心是一个 RNN 状态转移：$h_t = A h_{t-1} + B x_t$ 和 $y_t = C h_t$。
* 核心创新：选择性 (Selectivity) 。与 $A, B, C$ 固定的传统 SSM 不同，Mamba 的 $A, B, C$ 矩阵（以及离散化步长 $\Delta$）是输入依赖 (Input-dependent) 的。这意味着 Mamba 可以根据当前 $x_t$ 的内容，动态决定要从 $h_{t-1}$ 中保留什么、要从 $x_t$ 中吸收什么。
* 训练：Mamba 无法像 RetNet 那样推导出简单的并行矩阵形式。它依赖于一种硬件感知的并行扫描 (Parallel Scan / Prefix Sum) 算法，以并行方式计算其顺序状态。

RetNet (门控线性 RNN):
* 机制：如前所述，是一个线性 RNN：$S_n = \gamma S_{n-1} + K_n^\intercal V_n$ 39。
* 核心创新：固定的多尺度衰减。$\gamma$ 是按头固定的，不依赖于输入。所有的数据依赖性都来自 $Q, K, V$ 投影和外部的 Swish 门控。

核心差异：静态 vs. 动态状态转移这是 RetNet 和 Mamba 之间最根本的区别。

* Mamba 拥有一个动态的状态转移。它可以智能地“选择”何时遗忘、何时记忆，使其在需要复杂内容感知的任务（如“选择性复制”）上表现出色。
* RetNet 拥有一个静态的状态转移（衰减率 $\gamma$ 是固定的）。它的记忆衰减是预设的，而非动态学习的。这使其数学形式更简洁，并行/循环的对偶性更优雅，但可能在灵活性上不及 Mamba。

尽管机制不同，但 RetNet 和 Mamba 的成功共同指向了一个清晰的趋势：算法与硬件的协同设计（Hardware Co-Design）。无论是 RetNet 对融合核函数的需求，还是 Mamba 对并行扫描的依赖，都证明了新一代架构的成功不再仅仅是算法的成功，而是算法与底层硬件实现紧密耦合的胜利。

## 实现、应用与未来方向

Microsoft TorchScale RetNet 的官方实现包含在 Microsoft 的 torchscale 库中 。研究人员可以通过 RetNetConfig 定义模型参数，并使用 RetNetDecoder 实例化模型。Hugging Face 兼容性 对于 RetNet 的广泛采用而言，syncdoth/RetNet  等社区项目至关重要。这些项目提供了与 Hugging Face transformers 生态系统完全兼容的实现。


这意味着 RetNet 模型可以：   

1. 使用 Hugging Face 的 Trainer API 进行训练。
2. 通过 save_pretrained 和 from_pretrained 进行保存和加载。
3. 使用标准的 .generate() 方法进行文本生成。

这种兼容性极大地降低了研究人员和开发者尝试、微调和部署 RetNet 的门槛，是连接研究和实践的关键桥梁。

Frenos 公司的白皮书提供了一个 RetNet 在现实世界中应用的绝佳案例。网络安全（如威胁检测）需要对极长的事件日志序列（Streaming Data）进行实时分析。Transformer 在处理此类任务时，它架构的“处理时间呈二次方 ($O(N^2)$) 增长” ，使其无法满足实时性要求。而Frenos 的研究发现，“无论序列长度如何，RetNet 模型都表现出恒定的处理时间，凸显了其效率和优势” 。这个案例是对 RetNet 范式 2（循环表示） 的一次强有力的实践验证。对于任何需要处理流式数据、无限上下文（Infinite Context）或受资源限制的场景（如 IoT、实时监控），RetNet 的 $O(1)$ 恒定推理成本是 Transformer 架构无法比拟的根本性优势。

RetNet（保留网络）为解决序列建模中的“三难困境”提供了一个非常有前景的解决方案。它的核心创新在于“留存机制”和其数学上的对偶性，使得模型可以在训练时完全并行处理，而在推理时则转化为一种计算和内存效率都非常高（$O(1)$）的形式。

RetNet 提供了三种不同的计算模式：并行计算、循环计算和分块循环计算，这三种模式分别解决了训练时的并行性、推理时的低成本和处理长序列训练的挑战。并且，它在硬件使用方面非常高效，虽然它没有直接使用类似 FlashAttention 2 这样的技术，但它推动了硬件加速领域向更通用的融合框架（比如 AttentionEngine）发展，以适应各种架构的需求。

与另一种模型 Mamba 比较，RetNet 的机制相对简单，它采用固定衰减的静态状态转移，而 Mamba 则使用更灵活的动态机制。未来，正如业内所提到的，评估标准和基准数据集的统一是非常重要的，这有助于公平地比较 RetNet、Mamba 和其他 Transformer 替代方案在不同任务中的表现，从而决定谁能成为 Transformer 的真正继任者。

## 参考文献

1. Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2023). Retentive Network: A Successor to Transformer for Large Language Models.
2. Team, Z. (n.d.). RETNET: Challenging the Inference Cost of Transformer Based Language Models. Zontal.io. 
3. Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2022). Retentive Network: A Successor to Transformer for Large Language Models.
4. Khowaja, S. A. (n.d.). RetNet (The Transformer Killer) Demystified. Medium. 
5. SabrePC Blog. (n.d.). RNNs vs LSTM vs Transformers. SabrePC.com. 
6. (2023, August 2). RetNet to Reduce LLM Inference Cost[Video]. YouTube.
7. AI Fusion Labs. (2023, August 15). Retentive networks (RetNet) explained: The much-awaited transformer's killer is here. 
