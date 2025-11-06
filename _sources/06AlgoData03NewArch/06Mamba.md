<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# Mamba 与选择性状态空间模型：一项全面的技术综述

> Author by: 张嘉瑶


## S4的成功与内在局限

S4及其变体在多个长序列基准测试（如Long Range Arena）上取得了当时最先进的性能，首次证明了结构化SSM在这一领域能够超越Transformer。它成功地确立了SSM作为一种兼具卷积和循环双重优势的强大序列建模新范式。

这种双重性并非偶然的数学技巧，而是由DPLR结构所赋予的根本属性。DPLR是连接SSM两种计算视图的桥梁：它使得模型在训练时可以被看作一个可并行的卷积层，在推理时又可以被看作一个轻量级的循环单元。这优雅地解决了长期以来困扰序列模型设计的核心矛盾——RNN的推理效率与CNN/Transformer的训练效率之间的权衡。DPLR是S4最深刻的架构贡献，也为Mamba的诞生奠定了基础。

内在局限：
* 输入无关性： 这是S4最根本的局限。系统的核心矩阵 $(\overline{A}, \overline{B}, \overline{C})$ 以及由此产生的卷积核 $\overline{K}$，在模型训练完成后就被固定下来。它们的值不随输入序列的内容而改变。这意味着S4本质上是一个**线性时不变** **Linear Time-Invariant**系统。这种特性使其难以处理需要根据内容进行判断的任务，例如，从一段文本中有选择地复制某个特定信息，或者在处理信息时主动忽略无关的干扰项。
* 硬件效率问题： S4对FFT和特殊的柯西矩阵快速算法的依赖，在实践中可能成为性能瓶颈。这些算法在GPU上的实现并非总是最优化的，可能导致硬件利用率不高。


如下图的选择性复制任务中，我们希望复制输入的**部分内容**并按顺序输出：

<div align="center">
  <img src="../images/06AlgoData03NewArch/S4_Bad_Behavior1.gif" alt="img" />
</div>

然而，由于（循环/卷积）SSM 具有线性时不变性，因此它在此任务中表现不佳。正如我们之前所见，对于 SSM 生成的每个 token，矩阵A 、B和C都是相同的。

因此，SSM 无法进行内容感知推理，因为它会根据固定的 A、B 和 C 矩阵，平等地对待每个 token。无法做到选择性复制提示词。


SSM表现不佳的第二个任务是感应头，其目标是重现输入中发现的模式：

<div align="center">
  <img src="../images/06AlgoData03NewArch/S4_Bad_Behavior2.gif" alt="img" />
</div>

在上面的例子中，我们本质上是在进行一次性提示，试图“教”模型在每个“问： ”之后给出“答：”的答案。然而，由于SSM具有时不变性，它无法从历史记录中选择要调用的先前标记。

让我们通过关注矩阵B来说明这一点。无论输入x是什么，矩阵B都保持不变，因此与x无关：

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba1.jpg" alt="img" />
</div>

同样，无论输入如何， A和C也保持不变。这证明了我们迄今为止所见的 SSM 的静态特性。

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba2.jpg" alt="img" />
</div>

相比之下，这些任务对于 Transformer 来说相对容易，因为它们会根据输入序列动态地改变注意力。它们可以选择性地“查看”或“关注”序列的不同部分。SSM 在这些任务上的糟糕表现说明了时间不变 SSM 的根本问题，矩阵A 、B和C的静态性质导致了内容感知问题。


S4的这些局限性并非孤立的缺陷，它们之间存在着紧密的因果联系，并共同指向了下一代模型（即Mamba）所必须进行的创新方向。首先，最核心的概念缺陷是输入无关性。一个无法根据其处理内容来调整自身行为的模型，其能力必然是受限的。这是Mamba必须解决的“为什么”的问题。解决方案显而易见：让系统矩阵（如 $B, C$ 和步长 $\Delta$）成为输入的函数，从而构建一个时变系统。然而，这一改动直接破坏了SSM与静态卷积之间的等价性。一个时变系统无法用一个固定的卷积核来表示，这迫使模型回归到循环计算的范式。但标准的循环计算无法并行训练，这又会引入新的瓶颈。因此，最终的、也是最关键的创新，必须是一个硬件感知的、可并行的循环计算算法（即并行扫描），它既能高效地训练一个输入依赖的SSM，又能解决S4遗留的硬件效率问题，同时赋予模型至关重要的选择性能力。这条逻辑链清晰地将S4的问题直接映射到了Mamba的解决方案。

## Mamba范式——选择性状态空间的黎明

Mamba架构的出现，可以被视为对S4局限性的一次直接而精彩的回应。它通过引入选择性机制、硬件感知的并行算法和简化的整体架构，将SSM的能力提升到了一个新的高度。

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba3.jpg" alt="img" />
</div>

如上图所示，SSM通过将历史记录压缩成一个较小的状态，从而实现了高效的计算。这个方法能够减少计算开销，因为它不需要保存整个历史数据。然而，与Transformer通过注意力机制处理整个历史记录的方式相比，SSM的表达能力相对较弱。
Mamba的目标则是结合两者的优点。它通过保持较小的状态，同时也能实现与Transformer类似的强大功能，既保持了高效性，又能处理复杂的序列数据。

### 超越静态卷积核：从时不变性到输入依赖性的哲学转变

S4是一个线性时不变（LTI）系统，而Mamba是一个线性时变（LTV）系统。这一转变是两者最核心的区别。Mamba引入了**选择性**的核心思想，即模型应具备根据输入内容动态地增强或忽略信息的能力。为了实现这一点，Mamba将SSM的关键参数——步长 $\Delta$、输入矩阵 $B$ 和输出矩阵 $C$——设计为当前输入词元 $x_t$ 的函数。这意味着在序列的每一个时间步，SSM都会使用一套“定制”的参数。这种机制允许模型在每个时间步动态地调整其状态转换和信息流，从而实现对上下文的精细控制。


为了选择性地压缩信息，我们需要参数依赖于输入。为此，我们首先在训练期间探索 SSM 中输入和输出的维度：

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba4.jpg" alt="img" />
</div>

在结构化状态空间模型 (S4) 中，矩阵A 、B和C与输入无关，因为它们的维度N和D是静态的且不会改变。

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba5.jpg" alt="img" />
</div>

相反，Mamba 通过合并输入的序列长度和批量大小，使矩阵B和C，甚至步长∆依赖于输入：

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba6.jpg" alt="img" />
</div>

这意味着对于每个输入标记，我们现在有不同的B和C矩阵，从而解决了内容感知问题！

> 注意：矩阵A保持不变，因为我们希望状态本身保持静态，但其受影响的方式（通过B和C ）是动态的。

由于它们现在依赖于输入，因此它们会选择性地选择在隐藏状态下保留什么以及忽略什么。

较小的步长∆会导致忽略特定的单词，而是更多地使用先前的上下文，而较大的步长∆会更多地关注输入单词而不是上下文：

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba7.jpg" alt="img" />
</div>


综上所述为了选择性保留信息，Mamba做了以下改动：
* $\Delta$（步长）：它控制着模型在“记忆”和“遗忘”之间的平衡。一个较大的 $\Delta$ 会使得离散化的状态矩阵 $\overline{A}$ 的范数变小，从而更多地“遗忘”之前的状态 $x_{t-1}$，并更多地关注当前输入 $u_t$。相反，一个较小的 $\Delta$ 会让模型更多地保持和沿用旧的状态。通过动态调整 $\Delta$，模型可以决定在遇到关键信息时“重置”状态，或在处理连续信息时“维持”状态。
* $B$ 和 $C$（输入/输出投影）： 这两个矩阵由输入 $x_t$ 动态生成。这使得模型可以有选择性地决定将输入的哪些信息写入状态 $x_t$（由 $B$ 控制），以及从状态中读取哪些信息作为输出 $y_t$（由 $C$ 控制）。
* $A$（状态矩阵）： 为了保持系统的稳定性和结构性，核心的状态矩阵 $A$ 在Mamba中仍然是固定的（非输入依赖的），但其在离散化后的有效作用会受到输入依赖的 $\Delta$ 的动态调制。

Mamba的这种选择机制，实际上是一种以连续动态系统语言实现的“注意力”。传统注意力机制通过计算离散的softmax权重来聚合信息，而Mamba则通过由 $\Delta, B, C$ 共同调制的SSM状态的连续演化来控制信息在时间维度上的流动。当模型需要关注当前输入时，它可以通过一个大的 $\Delta$ 来“刷新”其内部状态；当需要回顾历史时，它可以通过小的 $\Delta$ 来维持状态。$B$ 和 $C$ 的投影则类似于注意力机制中的 Query, Key, Value 投影，决定了写入和读取信息的具体内容。因此，Mamba并非完全摒弃了注意力的核心思想（选择性），而是以一种更高效、更结构化的SSM框架重新实现了它。

### 架构深潜：Mamba模块

Mamba的整体架构块（Mamba Block）在设计上追求简洁与高效。它将核心的选择性SSM（SSM）与一个带有门控机制的类MLP结构相结合。输入信号 $x$ 首先经过一个线性投影，然后被分成两路：一路进入SSM进行序列状态的演化，另一路进入一个带有SiLU（Swish）激活函数的门控MLP。最后，SSM的输出与门控MLP的另一路输出相乘，形成最终的模块输出。这种结构比标准的Transformer块更简单，因为它移除了多头注意力机制和块内复杂的层归一化（Layer Normalization）操作。

在机器学习的骨干网络中，通常包含两个基本操作：token 间的信息通信(Communication)和 token 内的计算(Computation)$^{6}$。Mamba 采用 SSM 进行通信(取代了注意力机制)，并保留了类似 MLP 的投影进行计算 $^{6}$。一个典型的 Mamba 模块通常包括：一个输入线性投影层、一个一维卷积层(Conv1D)、SiLU 激活函数、选择性状态空间模型(S6)核心以及一个输出投影层。此外，通常还带有一个残差连接和一个门控机制(通过与另一个经过 SiLU 和线性投影处理的分支进行逐元素相乘实现)

Mamba 模型具有多个接口级别，但其核心是封装了选择性 SSM 的 Mamba

下图展示了一个 Mamba 模块的概念框图：

<div align="center">
  <img src="../images/06AlgoData03NewArch/MambaBlock.png" alt="img" />
</div>




### 硬件感知算法：为GPU性能融合内核操作

使Mamba的动态选择性在实践中可行的，是其在算法层面的关键创新。由于参数是输入依赖的，S4所依赖的静态卷积技巧不再适用，模型必须依赖循环计算。然而，朴素的循环计算在GPU这类并行硬件上效率极低，因为它无法利用大规模并行处理单元。

如图所示，每个状态都是前一个状态（乘以A ）加上当前输入（乘以B ）之和。这称为扫描操作，可以使用 for 循环轻松计算。相比之下，并行化似乎是不可能的，因为每个状态的计算都只能在已知前一个状态的前提下进行。


<div align="center">
  <img src="../images/06AlgoData03NewArch/MambaRNN.gif" alt="img" />
</div>


Mamba的解决方案是设计了一种**并行扫描**算法。该算法在数学上与顺序的循环计算完全等价，但其计算结构被重塑，使其能够高效地在现代硬件上并行执行。它将一个看似顺序的计算链分解为多个可以并行处理的块，然后在块之间进行少量的顺序聚合。


<div align="center">
  <img src="../images/06AlgoData03NewArch/MambaGPU.png" alt="img" />
</div>

但GPU的一个缺点是其容量较小但效率极高的 SRAM 与容量较大但效率略低的 DRAM 之间的传输 (IO) 速度有限。频繁在 SRAM 和 DRAM 之间复制信息会成为瓶颈。

<div align="center">
  <img src="../images/06AlgoData03NewArch/GPUBottleNeck.png" alt="img" />
</div>


而Mamba算法是“硬件感知”的。Mamba试图将离散化、循环更新和输出投影等多个计算步骤**融合** **fuse**到单个GPU内核中。这样做可以最大限度地减少数据在GPU不同层级内存（高带宽内存HBM与片上高速SRAM）之间的读写次数。通过避免在每一步都将完整的中间状态 $x_t$ 写回全局内存，Mamba极大地提升了计算速度并降低了内存占用，从而实现了卓越的硬件利用率。


<div align="center">
  <img src="../images/06AlgoData03NewArch/MambaGPUPro.gif" alt="img" />
</div>

我们可以通过可视化 Mamba 的基础架构来查看 DRAM 和 SRAM 分配的具体实例：

<div align="center">
  <img src="../images/06AlgoData03NewArch/Mamba8.png" alt="img" />
</div>

这里，以下内容融合成一个内核：

* 离散化步骤，步长为$∆$

* 选择性扫描算法

* 用C进行乘法

硬件感知算法的最后一部分是重新计算。

中间状态不会被保存，但对于反向传播计算梯度来说却是必需的。因此，作者在反向传播过程中重新计算了这些中间状态。

尽管这看起来效率不高，但比从相对较慢的 DRAM 读取所有中间状态的成本要低得多。

现在，我们已经介绍了其架构的所有组件，该架构使用文章中的下图进行描述：


<div align="center">
  <img src="../images/06AlgoData03NewArch/MambaArch.png" alt="img" />
</div>


这种算法与硬件的协同设计是Mamba成功的核心。选择性SSM的理论模型若没有并行扫描算法的支持，将因无法高效训练而变得不切实际；而并行扫描算法若没有选择性SSM这一应用场景，也只是一个通用的计算技巧。这标志着AI研究的一个新前沿：算法创新与系统级工程优化必须紧密结合、同步进行。未来的架构突破可能越来越需要跨越机器学习理论、计算机体系结构和系统编程等多个领域的深度专业知识。


凭借选择性机制和硬件感知算法，Mamba成功地解决了序列建模领域长期存在的“循环-并行”权衡：
* 训练时： Mamba利用并行扫描算法，使其本质上是循环的结构能够以接近线性的时间复杂度（$O(L \cdot D \cdot N)$）和极高的硬件利用率进行训练，效率媲美Transformer。
* 推理时： Mamba可以切换回其原始的、高效的顺序循环模式。每生成一个新词元，只需进行一次 $O(D \cdot N)$ 的状态更新，计算和内存成本均为常数 $O(1)$，使其在自回归生成任务中速度极快且内存占用极低。

## 架构对比分析

S4可以被视为一个里程碑式的“概念验证”，它首次证明了结构化SSM作为高效长序列模型的可行性，并建立了其卷积/循环的双重计算模式。而Mamba则是其“精炼的继任者”，它精准地解决了S4的根本缺陷（输入无关性）和实践瓶颈（硬件利用率）。

从S4到Mamba的进化，核心是从一个线性时不变（LTI）系统到一个线性时变（LTV）系统的飞跃。这一飞跃之所以能够成功，关键在于Mamba通过算法与硬件的协同设计（并行扫描），使得这个更强大、更灵活的时变模型在计算上依然是可行的。


### Mamba vs. Transformer：复杂度、性能与归纳偏置的正面对决

Mamba与Transformer在多个维度上展现了根本性的差异，这些差异可以通过下表进行系统性地总结和分析。


| 特征 | RNN/LSTM | Transformer (Attention) | S4 (结构化SSM) | Mamba (选择性SSM) |
|------|----------|-------------------------|----------------|------------------|
| **训练复杂度** | $O(L\cdot D^{2})$ (顺序) | $O(L^{2}\cdot D)$ (平方) | $\tilde{O}(L(D+N\log N))$ (近线性) | $O(L\cdot D\cdot N)$ (近线性) |
| **自回归推理** | $O(D^{2})$ (每步恒定) | $O(L\cdot D)$ (上下文线性) | $O(D\cdot N)$ (每步恒定) | $O(D\cdot N)$ (每步恒定) |
| **训练并行性** | 否 (顺序依赖) | 是 (完全并行) | 是 (通过卷积/FFT) | 是 (通过并行扫描) |
| **长距离依赖处理** | 差 (梯度消失) | 优 (直接路径) | 优 (HiPPO结构) | 优 (HiPPO+选择性) |
| **输入依赖动态** | 是 (非线性门控) | 是 (自注意力) | 否 (线性时不变) | 是 (选择性参数) |
| **推理内存占用** | 恒定 | $O(L\cdot D)$ (KV缓存) | 恒定 | 恒定 |


从上表中可以清晰地看到，Mamba在多个关键指标上实现了对以往架构的超越。复杂度： Mamba在训练时实现了与序列长度 $L$ 的线性扩展（$O(L)$），彻底摆脱了Transformer的平方复杂度（$O(L^2)$）瓶颈。在推理时，它恢复了RNN的优势，每步计算成本为常数，而Transformer则需要随着上下文的增长消耗更多的计算和内存（KV缓存）。性能： 线性扩展的能力使得Mamba能够轻松处理百万级长度的序列，这是标准Transformer架构望尘莫及的。归纳偏置： 如前所述，Transformer作为关系型模型，而Mamba作为动态系统模型，后者可能更适合对连续信号进行建模。此外，Mamba没有固定的上下文窗口限制，其状态可以理论上无限地传递信息。该表格直观地展示了Mamba如何集两家之长：它既拥有Transformer和CNN的训练并行性，又具备RNN的推理高效性，最终在“训练并行性”和“恒定推理成本”两个关键特性上都取得了肯定的答案。


### 经验证据：基准测试表现回顾

在实际的基准测试中，Mamba的表现印证了其架构设计的优越性。在语言建模任务上，Mamba不仅在性能上能够匹配甚至超越同等规模的Transformer模型，而且实现了更高的训练和推理吞吐量。在处理基因组学、音频和时间序列等典型的长序列数据模态时，Mamba展现出了压倒性的优势，成为这些领域的首选架构之一。

从在 HG38 人类基因组数据集上的预训练结果来看，Mamba 在处理基因组学任务时的核心优势可归纳为以下两点，分别对应左图和右图的实验结论：
* 参数规模扩展性更优：小参数量下仍能高效提升性能
* 长序列处理能力更强：上下文长度增加时性能不衰减（甚至提升）

<div align="center">
  <img src="../images/06AlgoData03NewArch/MambaDNA.jpg" alt="img" />
</div>


这对基因组学任务至关重要：人类基因组序列极长（总长约 30 亿碱基对），且许多关键生物学信息（如长程基因调控、跨片段突变关联）隐藏在长序列中。传统基线模型（如 Transformer 类）因自注意力机制的计算复杂度随序列长度平方增长，在处理长基因组片段时易出现性能瓶颈（如信息稀释、计算过载）；而 Mamba 的选择机制能高效聚焦长序列中的关键信息（如特定碱基模式、基因片段关联），更适配基因组学对 “长序列语义理解” 的核心需求。



### 参考文献


1. Gu, Albert, and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces." arXiv preprint arXiv:2312.00752 (2023).

2. Gu, Albert, et al. "Combining recurrent, convolutional, and continuous-time models with linear state space layers." Advances in neural information processing systems 34 (2021): 572-585.

3. Gu, Albert, et al. "Hippo: Recurrent memory with optimal polynomial projections." Advances in neural information processing systems 33 (2020): 1474-1487.

4. Voelker, Aaron, Ivana Kajić, and Chris Eliasmith. "Legendre memory units: Continuous-time representation in recurrent neural networks." Advances in neural information processing systems 32 (2019).

5. Gu, Albert, Karan Goel, and Christopher Ré. "Efficiently modeling long sequences with structured state spaces." arXiv preprint arXiv:2111.00396 (2021).

6. Grootendorst, M. (2023, October 29). A visual guide to Mamba and state space models. Maarten Grootendorst's Newsletter. https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
