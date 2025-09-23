<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 09. 流水并行 1F1B 及衍生算法

> Author by：高亮

在大模型训练中，并行技术是突破硬件算力与显存瓶颈的核心手段，而 PP 凭借对模型层的纵向切分能力，成为大规模训练的关键支柱。传统 GPipe 虽通过 micro-batch 切分摊薄气泡，却面临尾部气泡先天存在、激活显存随微批数 m 线性增长的难题，当 m 受硬件限制时，流水线深度增加反而会降低吞吐效率。

为突破这一困境，1F1B 即 PipeDream 应运而生，通过让 micro-batch 反向计算尽早折返并与后续前向交错，将激活驻留时间从 O(m) 压缩至 O(p)，大幅降低显存压力。在此基础上，Virtual Pipeline、PipeDream-2BW、ZB-V 等衍生技术进一步优化气泡率与通信开销。本文将从原理出发，拆解这些主流流水并行技术的设计逻辑与优势，为大模型高效并行训练提供技术参考。

## PipeDream 基本原理

### “两段式”流水线并行存在问题：

先全前向、再全反向的两段式调度流水线技术，以朴素流水和 Gpipe 为代表，参见[图 1](#fig1)。朴素流水相当于 $m{=}1$，GPipe 则把 mini-batch 切成 $m$ 个 micro-batch 并在前/反两个阶段各自成流水。两段式流水调度的核心问题在于结构性气泡不可消除、只能用 $m$ 摊薄：总空泡时间为 $t_{\text{bubble}}=(p-1)(t_f+t_b)$，理想计算时间为 $t_{\text{ideal}}=m(t_f+t_b)$，因此空泡率 $\frac{p-1}{m+p-1}$，利用率 $U=\frac{m}{m+p-1}$。当 $m$ 由于硬件内存有限而受限时，即使加深流水线 $p$ 也会让头尾气泡相对占比上升，吞吐反而不增。此外，两段式流水调度还存在以下问题：

- **尾部气泡先天存在**。因为前/反向两个阶段被硬性分隔，反向无法与后续前向交错，cooldown 的尾巴完全暴露，最多只能靠 $m$ 变大来“摊薄”。
- **通信–计算重叠受限**。前/反向阶段内可以把激活/梯度传输与算子计算重叠，但跨阶段无法把反向藏进下一轮前向，整体关键路径仍包含两段各自的 warmup/cooldown 开销，单步时延为 $(m+p-1)(t_f) + (m+p-1)(t_b)$。
- **显存与 $m$ 线性冲突**。为等反向传输、stage 必须同时长期保留本段 **$m$** 份前向激活，峰值近似 $M^{(i)}_{\text{act,peak}}\!\approx m\,L_i A_{\text{layer}}$，从而限制 $ m $ 的上限。

$$
 M_{\text{peak}} \;\approx\; \underbrace{P_{\max}}_{\text{参数系}} \;+\; \underbrace{\max_i \big(m L_i A_{\text{layer}}\big)}_{\text{激活系}}\;\;\propto\;\; \text{Params} + \text{Activations}\times m.
$$

- **微批切分的边际收益递减**。增大 $m$ 还会引入更密的启动/同步开销与更小的 per-kernel 批量，可能抵消部分吞吐收益。综合来看，“两段式”PP 的瓶颈是：要效率就要大 $m$，但大 $m$ 又受激活显存与系统长尾约束——这正是后续更先进调度试图突破的根因。

<a id="fig1"></a>
![两段式流水原理](../images/04Train02ParallelAdv/10pipeline01.png)
**图 1** 两段式流水原理

### 流水并行 1F1B：PipeDream

1F1B 的核心是让每个 micro-batch 的反向尽早折返并与后续前向交错。与两段式：先全前向、再全反调度过程不同，1F1B 在 warmup 完成后直接进入稳态：末段 stage 一旦完成某个 micro-batch 的前向，就立即启动该 micro-batch 的反向（因此为 1F1B 即 one-forward-one-backward）；与此同时，前端 stage 继续为后续 micro-batch 做前向，如[图 2](#fig2)所示。这样，反向梯度沿着流水线向前回传，各 stage 在时间线上呈现出 F, B, F, B… 的交替节奏，仅在开头/结尾保留不可避免的 warmup/cooldown。由于反向被尽早触发，同一份激活从“产生到被消耗”的距离由 GPipe 的 $O(m)$（要等全前向结束）降为 $O(p)$（只需等反向折返到本段），显著缩短驻留时间。

>理解： “激活从产出到被消耗的距离”理解为：该激活在内存里需要等待多少个流水“时隙（slot）” 才会被本段的反向读取并释放。对比 GPipe（两段式）与 1F1B（交错式），这个等待里是否包含“要等完全部 $m$ 个 micro-batch 的前向”是关键差异。设单个 stage 的前/反向各占 1 个时隙（便于计数），第 $i$ 段在前向处理第 $j$ 个 micro-batch 时产生活动张量。
> * **GPipe（先全前向、再全反向）**
  这份激活要等所有前向都做完，反向阶段才开始；随后还要等反向从末段折返到第 $i$ 段，并轮到编号 $j$ 的样本。等待时隙近似：
  $$
  \Delta t_{\text{GPipe}}(i,j)\;\approx\;
  \underbrace{(m+p-1)}_{\text{等前向阶段结束}}
  +\underbrace{(p-1-i)}_{\text{反向回传到第 }i\text{ 段}}
  +\underbrace{(m-1-j)}_{\text{倒序轮到第 }j\text{ 个样本}}
  \;-\;\underbrace{(i+j)}_{\text{激活产生时刻}}
  \;=\;\mathcal{O}(m)\;+\;\mathcal{O}(p).
  $$
  核心在第一项：必须先等完全部 $m$ 个 micro-batch 的前向，因此主导量级是 $\mathcal{O}(m)$。
> * **1F1B（One-Forward–One-Backward）**
  末段对第 $j$ 个样本一做完前向就立刻启动该样本的反向，不再等待其它样本的前向完成；反向只需沿流水深度折返到第 $i$ 段即可：
  $$
  \Delta t_{\text{1F1B}}(i,j)\;\approx\;
  \underbrace{(p-1)}_{\text{等末段拿到该样本}}
  +\underbrace{(p-1-i)}_{\text{反向回传到第 }i\text{ 段}}
  \;-\;\underbrace{i}_{\text{激活产生时刻的相位}}
  \;=\;\mathcal{O}(p).
  $$
  没有“等完整个 $m$”这项，量级与流水深度 $p$ 有关，而与 $m$ 无关。

> 若取 $p=4,\;m=8$，看 $ i=0 $ 段、样本 $j=0$ 的激活，对于 GPipe 来说，时延为：$\Delta t_{\text{GPipe}}\approx (8+4-1)+(4-1-0)+(8-1-0)-(0+0)=21$ 个时隙，随 $m$ 增大而变长。对于 1F1B 来说，时延为：$\Delta t_{\text{1F1B}}\approx (4-1)+(4-1-0)-0=6$ 个时隙，与 $m$ 无关。GPipe 单段需同时保留约 $m$ 份激活（峰值 $\propto m$）。1F1B 单段保留约与流水深度相关的少量份数（峰值 $\propto p$）。这就是“由 $\mathcal{O}(m)$ 降到 $\mathcal{O}(p)$，驻留时间缩短，从而激活峰值显存显著下降”的精确含义。


这一调度直接改写了动态峰值内存的量纲：若不做重计算（checkpointing 关闭），GPipe 的第 $i$ 段在前向结束时需同时保留 $m$ 份可反向激活，峰值近似 $m\,L_i A_{\text{layer}}$；而在 1F1B 的稳态交错下，每段同时“挂起”的在途 micro-batch 数量受流水深度限制，记常数 $k_i\!\sim\!O(p)$，峰值近似

$$
M^{(i)}_{\text{act, peak}} \approx k_i \cdot L_i \cdot A_{\text{layer}}
\quad\text{（无重算）},
$$

若开启梯度检查点，仅长期保留段边界激活，则

$$
M^{(i)}_{\text{act, peak}} \approx k_i \cdot A_{\text{boundary}} + \alpha\,L_i A_{\text{layer}}
\quad\text{（有重算）}.
$$

对比可见，1F1B 将激活峰值从 $\mathcal{O}(m)$ 压到 $\mathcal{O}(p)$，这正是其“先天降低显存压力”的根因：即使在相同 $m$ 下更省显存，或者在相同显存下允许把 $m$ 调得更大，从而进一步摊薄气泡、提升吞吐。

但就“气泡”本身而言，定义 $t_{\text{bubble}}=(p-1)(t_f+t_b)$、$t_{\text{ideal}}=m(t_f+t_b)$，则 bubble radio 保持不变，即：

$$
\mathit{bubble\ ratio}
=\frac{t_{\text{bubble}}}{t_{\text{bubble}}+t_{\text{ideal}}}
=\frac{p-1}{m+p-1}.
$$

这里 $p$ 为流水线深度（stage 数），$m$ 为 micro-batch 数，$t_f,t_b$ 为单个 micro-batch 在单个 stage 的前/反向时间。关键在于 $t_{\text{bubble}}$ 的来源与调度无关：不管是两段式（GPipe）还是交错式（1F1B），都必须经历 $ (p−1) $ 个“warmup/cooldown”的结构性头尾时延；这部分是由有限流水深度决定的不可消除开销，与是否交错无关（即 1F1B）。

即 1F1B 改善的是内存，而非空泡大小本身。1F1B 把反向尽早折返并与后续前向交错，使单份激活的驻留距离从 $O(m)$（需等全前向）降到 $O(p)$（只等反向折返），从而将峰值激活显存从 $\mathcal{O}(m)$ 压到 $\mathcal{O}(p)$。这并不会改变上式中的 $(p-1)$ 头尾时隙，也就不改变空泡率；但它显著降低峰值显存，允许在相同显存预算下把 $m$ 开得更大，于是 $ bubble\ radio$ 可以通过更大的 $m$ 被进一步压低——这是 1F1B 间接降低空泡占比、提升稳态吞吐的根本机制。

<a id="fig2"></a>
![PipeDream 原理](../images/04Train02ParallelAdv/10pipeline02.png)
**图 2** 1F1B 流水原理

## Virtual pipeline 基本原理


后续 Megatron-LM 在 1F1B 的基础上提出 Interleaved 1F1B，即虚拟流水并行（Virtual Pipeline Parallelism，VPP），用于进一步削减流水线气泡。其核心并非简单的通过增大 $ m $，即流水并行的数量来将流水线划分的更细，降低气泡占比，而是在每张物理 GPU 上引入 $v$ 个“虚拟流水阶段”（virtual pipeline stages）并交错调度（interleaving）：

如[图 3](#fig3)所示，不同于之前的 Pipeline 方案，一台 GPU 设备只承担一个或几个连续流水线阶段的计算任务，这导致了其它 GPU 存在等待数据的情况，因此采用了虚拟化的方法将多个不相邻阶段的流水线计算由同一 GPU 来承担，并通过多个并行线程或 CUDA 流在同一 GPU 上交错进行不同阶段的前向、反向计算，这样就实现了 GPU 在等待上/下游数据的空隙中能切换到本卡的其他虚拟阶段继续计算，充分利用了原本的空等（idle）时间。

<a id="fig3"></a>
![VirtualPP 原理](../images/04Train02ParallelAdv/10pipeline04.png)
**图 3** VPP 原理

### 虚拟化的理解

给定总层数 $L_{\text{total}}$、流水线并行度 $p$（物理设备数）、虚拟并行度 $v$（每卡虚拟阶段数），将模型划分为：

$$
p\times v \quad \text{个连续、等长的层段，段长度} \; L_{\text{seg}}=\frac{L_{\text{total}}}{p\,v}.
$$

对第 $d\in\{0,\dots,p-1\}$ 张 GPU，分配其虚拟阶段索引：

$$
\mathcal{S}_d=\{\,s\mid s\equiv d \ (\mathrm{mod}\ p),\ s\in[0,pv-1]\}.
$$

虚拟化的对象为流水阶段，因此同一 GPU 负责 $v$ 个不相邻的阶段（跨步映射）。执行时，为 $\mathcal{S}_d$ 中的每个虚拟阶段各开一条 CUDA 流，在 1F1B 语义下交错推进这些更小的前/反片段：当某虚拟阶段因 P2P 传输/上游依赖而空等时，GPU 立即切换到本卡的另一虚拟阶段计算，以此填充原本等待的时隙。

<a id="fig4"></a>
![原理](../images/04Train02ParallelAdv/10pipeline03.png)
**图 4** 虚拟化解释


### 降低空泡率
在 1F1B 中，warmup/cooldown 需要跨越 $(p-1)$ 个总量不变的流水线阶段（因为 $ p $ 恒定，即模型的纵向切分数目确定，总的流水线阶段数目确定）。VPP 并不减少流水线阶段总数，而是把“每次跨越单台设备对应的流水线阶段”按 $1/v$ 缩短（因为单台设备流水线阶段不相邻，若之前跨越单个 GPU 需要跨越 4 个流水线阶段，VPP 仅需跨越 1 个阶段就可以进入下一个 GPU 进行下一阶段流水线计算，[图 4](#fig3)清晰的展示了这一过程）：因此给定不开启 VPP 情况下前/反向时间分别为 $t_f, t_b$，则开启 VPP 后的前/反向时间为：

$$
t_f^{(v)}=\frac{t_f}{v},\qquad t_b^{(v)}=\frac{t_b}{v}.
$$

于是同样 $(p-1)$ 次跨越的 bubble 时间压缩为

$$
t_{\text{bubble}}^{\text{VPP}}
=(p-1)\Bigl(t_f^{(v)}+t_b^{(v)}\Bigr)
=\frac{(p-1)(t_f+t_b)}{v}.
$$

理想工作量不变（$t_{\text{ideal}}=m(t_f+t_b)$），因此空泡率

$$
\text{bubble ratio}^{\text{VPP}}
=\frac{t_{\text{bubble}}^{\text{VPP}}}{t_{\text{ideal}}}
=\frac{\frac{(p-1)(t_f+t_b)}{v}}{m(t_f+t_b)}
=\frac{1}{v}\cdot\frac{p-1}{m}.
$$

> 直观解释：必须经历的流水线阶段数量仍是 $(p-1)$ 次，但每次只推进 $1/v$ 的前/反向片段，因而总等待时隙宽度按 $1/v$ 缩减；理想算量不变，故空泡率与 $v$ 成反比。

**总结**
VPP 的“虚拟化”不是把多张 GPU 合并，而是在每张 GPU 内开出 $v$ 个虚拟流水阶段，并行/交错推动更小的前/反片段以填补时隙。其结果是：在不改变总算量与 1F1B 低驻留优势的前提下，将气泡时间缩小为 $\tfrac{1}{v}$ 倍（$\text{bubble ratio}^{\text{VPP}}=\tfrac{1}{v}\cdot\tfrac{p-1}{m}$），以通信频率 $\times v$ 的代价换取更小的气泡和更高的稳态利用率。

### VPP 的通信开销：

通信开销体现在两个方面：通信频率的提升和通信容量的提升。

相比于非 VPP 流水线技术，由于每张 GPU 承载了 $v$ 个虚拟流水线阶段，micro-batch 的需要先经过每台设备的第一个虚拟阶段，再循环经过每台设备的第二个虚拟阶段……因此单个 micro-batch 需要以 1F1B 的方式穿过共计 $p\times v$ 个流水线阶段，对于非 VPP 来说仅需穿过 $p$ 个流水线阶段即可。与非 VPP 技术相比，由于 VPP 将相邻的流水线阶段落在不同的 GPU 上，因此在模型前向训练过程的跨 GPU 跃迁次数由之前的 $p-1$ 次增加到 $vp-1$ 次，对于反向过程同理。

若设跨越 GPU 通信的激活量大小为 $S$ 字节（梯度大小近似同阶），则：
* **非 VPP（$v=1$）**：

  * 前向全路径字节：$(p-1)\,S$
  * 反向全路径字节：$(p-1)\,S$
  * 合计：$(p-1)\,2S$ / micro-batch
* **VPP（$v>1$）**：

  * 前向：$(p\,v-1)\,S \approx v(p-1)S$
  * 反向：同上
  * 合计：$v\,(p-1)\,2S$ / micro-batch
> 结论：与非 VPP（$v{=}1$）相比，通信频率 $\times v$，总字节量 $\times v$。


## 新兴 PP 技术（扩充）


### PipeDream-2BW

![PipeDream-2BW 原理](../images/04Train02ParallelAdv/10pipeline05.png)


#### 核心思想：

目标：在不做周期性 flush 的前提下，兼顾高吞吐与低显存，同时尽量贴近数据并行的更新语义。做法是把 1F1B 调度与双缓冲权重（2-Buffered Weights, 2BW）和梯度合并（coalescing）结合起来，即每个 stage 只保留两份权重版本（current / shadow），而不是像原始 PipeDream 可能需要最多 $d$ 份；同时对同一 micro-batch 的前/反向严格用同一份权重（消除“前后不一致”），但权重更新采用 1-stale 语义，避免 flush 引起的停顿。

#### 调度：

* 仍按 1F1B 调度：各 stage 交替执行不同 micro-batch 的前向、反向过程。
* 设每批累计 $m$ 个 micro-batch（梯度在批内合并），每处理完 $m$ 个 micro-batch 产出一个新权重版本，且要求 $m\ge d$（流水深度），常见简化是 $m=d$。新版本只供新进入管线的样本使用，管线中尚在飞行的样本继续用其前向时那份版本完成反向，因此每个 stage 最多只需两份版本（current + shadow）。

> 直观：版本推进是“批”为单位滚动的；“老版本”一旦其对应的 in-flight micro-batch 反向都消化完，就可丢弃，内存占用上限因此被钉在 2 份。

#### 更新语义（1-stale）与优化器

把批级权重记作 $W(t)$，批平均梯度为 $\nabla f(W)$。

* **标准小批 SGD**：$W(t{+}1)=W(t)-\nu\,\nabla f(W(t))$。
* **2BW（1-stale）**：$\boxed{W(t{+}1)=W(t)-\nu\,\nabla f\big(W(t{-}1)\big)}$。
  延迟常数为 **1**，对所有 stage 一致；实验显示与 vanilla 语义**收敛相当**。动量/Adam 等也可平移到“1-stale 梯度”上，无需额外影子变量。

#### 显存与吞吐：

* **显存**：

  * 权重版本：2 份（2BW） vs 最多 $d$ 份（PipeDream），显存大幅下降；
  * 激活：仅为 in-flight micro-batch 的缓存（随 1F1B 为 $O(p)$ 量级），远小于两段式在大 $m$ 下的 $O(m)$ 驻留。
* **吞吐**：无 flush 的结构性空转，稳态效率显著高于 GPipe；论文报告对 GPT/BERT 类模型可比优化基线加速 1.3×–20×、较 GPipe 可达 3.2×。

#### 和常见基线对比：

| 技术                  | 调度               | 权重版本  | 语义            | 典型特征                                  |
| ------------------- | ---------------- | ----- | ------------- | ------------------------------------- |
| **GPipe**           | 两段式 | 1     | vanilla       | 需周期性 flush；空泡显著；激活驻留 $\propto m$。 |
| **PipeDream（原版）**   | 1F1B（无 flush）    | ≤ $d$ | 多步 stale（不均匀） | 吞吐高但权重版本多、显存高、staleness 难控。           |
| **PipeDream-Flush** | 1F1B（有 flush）    | 1     | vanilla       | 显存更低，但因 flush 吞吐下降。                   |
| **PipeDream-2BW**   | 1F1B（无 flush）    | **2** | **1-stale**   | 兼顾高吞吐与低显存；收敛与 vanilla 接近。             |


总结：2BW = 1F1B + 两份权重 + 批内合并 + 1-stale 更新：把 PipeDream 的多版本与不均匀陈旧度收敛成“每段最多两份、全段统一 1-stale”，在不 flush 的前提下做到高吞吐、低显存、收敛几乎等同数据并行。

### ZB-V schedule

![ZB-V schedule 原理](../images/04Train02ParallelAdv/10pipeline06.png)

#### 核心思想：

ZB 系列的关键创新是把反向传播拆成两部分：B<sub>in</sub>：对输入的梯度（把梯度往上游传）；B<sub>w</sub>：对权重的梯度（局部权重的 dW 计算）。 利用这两段可“自由插入”的反向子片段去填充流水线中原本不可避免的空隙，从而在同步训练语义下把气泡压到接近 0。ZB-V 是其中一类手工设计的日程：每张 GPU 负责 2 个模型分块（chunk），前向与两段反向的依赖在时间轴上呈 “V” 字形，因此得名。

#### V 字形调度：

* 将每个物理 stage 再细分为 两个 chunk，并安排它们在时间线上以 F → B<sub>in</sub> / B<sub>w</sub> 交错的方式执行；不同设备上的两个 chunk 彼此错位，让一个 chunk 的 B<sub>w</sub> 能“卡位”填上另一个 chunk 上的空隙。
* 稳态下，前向（F）与两段反向（B<sub>in</sub>、B<sub>w</sub>）交替推进，把原本 1F1B 尾部 cooldown 暴露的空隙用 B<sub>w</sub> 塞满，因此“零气泡”的条件变成：F、B<sub>in</sub>、B<sub>w</sub> 的时长足够匹配，否则仍会留下少量残余空隙（需用贪心/自动调度补偿）。

> 直观理解：相比 1F1B 只有“F 与 B”两种拼块，ZB-V 有了“F、B<sub>in</sub>、B<sub>w</sub>”三种拼块，可在时间线上更细粒度地“打补丁”。当三者时长接近时，补丁正好把缝隙补平，气泡≈0。

#### 何时能做到“Zero-Bubble”

* **理想条件**：若单 chunk 的 F ≈ B<sub>in</sub> ≈ B<sub>w</sub>，ZB-V 能实现“零气泡”属性（同步训练语义下）。实际模型不完全等时，需引入贪心/搜索调度以靠近零气泡。
* **优化器屏障**：论文还提出通过绕过优化器步的同步进一步消除尾部屏障，这是达成“真正零气泡”的工程要点之一。

#### 内存影响（与 1F1B/可控内存的关系）

* **激活峰值**：ZB 的设计可在不抬升 1F1B 峰值的情况下显著减泡；若把零气泡作为硬约束，通常需要更高的激活占用（文献报告“接近零气泡”在现实设定下往往需要 \~2× 1F1B 的激活预算；而在理想均衡下最低可到 \~1/2 的参数-激活校准内存）。
* **权重与版本**：ZB-V 不必像 PipeDream 那样存很多权重版本；其重点在于切分 B<sub>in</sub>/B<sub>w</sub> 的算子级时序，在 1F1B 的低驻留量纲上（≈O(p)）做更精细的时间编排。

#### 通信/实现代价

* **通信条数**：与 VPP 相似，更细粒度的片段意味着更多边界消息；B<sub>in</sub>、B<sub>w</sub> 的交错也会引入额外的依赖与消息序列。需要通过多 stream + NCCL 多通道 + 分块来隐藏通信。
* **复杂度**：需要调度器管理三类片段（F / B<sub>in</sub> / B<sub>w</sub>）的事件依赖与“何时插补”策略；官方实现已在 Megatron-LM 分支开源（含通用运行时与不同 ZB 族 schedule）。([GitHub][4])


#### 与 1F1B / VPP 的对比：

* **对 1F1B**：在保持 1F1B 低驻留（≈O(p)）的前提下，通过 B<sub>in</sub>/B<sub>w</sub>的三片段交错进一步吃掉空隙；当三者等时，理论上可达零气泡。
* **对 VPP**：VPP 用“更多虚拟阶段”把时隙宽度缩小到 $1/v$，气泡率随 $1/v$ 下降但通信×v；ZB-V 不依赖增加虚拟阶段，而是靠反向拆分与插补来减泡，属于不同维度的改进（两者可叠加，但需评估通信与内存的联合作用）。


总结：ZB-V 通过把反向拆成 B<sub>in</sub>/B<sub>w</sub> 并以 “V 字形”两块/卡 的方式交错插补，使前/反片段更细粒度地占满时间轴；在 F≈B<sub>in</sub>≈B<sub>w</sub>的条件下可实现近乎零气泡，同时维持接近 1F1B 的激活驻留量纲，但带来更复杂的调度与通信序列，落地时需依赖成熟的开源实现与周到的通信/均衡优化。在类似内存预算下，ZB-V 相对 1F1B 的吞吐提升可达 \~15–23%；在放宽内存时可到\~30%，体现其“以更细粒度时序填补空隙”的优势。


### Hanayo wave-like pipeline

![Hanayo wave-like 原理](../images/04Train02ParallelAdv/10pipeline07.png)


#### 核心思路：
将阶段数 $S$ 与设备数 $P$ 解耦，把模型切成比设备更多的阶段，并按“波（wave）”推进；一轮前后向中的波数定义为

  $$
  W=\frac{S}{2P}.
  $$

  当 $W>1$ 时，同一批微样本会像“波峰”一样在更细的阶段序列上连续推进，形成 wave-like 时序。这样能在不复制模型的前提下，进一步细化时隙、压缩气泡。统一框架：Hanayo 提供可表达主流流水方案的运行时，使用 action list 驱动执行，并配合异步预取（prefetch）把通信尽量叠在计算后侧。

#### 气泡/吞吐

对 wave-like 时序的理论气泡模型（单轮前后向）给出总时长形式（含前/反向与通信三部分）：

$$
T_{\text{iter}} \;=\; 
\underbrace{\frac{P-1}{P}\,T_F}_{\text{F 的结构性等待}}
+\underbrace{\Bigl(\frac{1}{2W}+\frac{1}{P}\cdot\frac{P}{P-1}\Bigr)T_B}_{\text{B 的结构性等待}}
+\underbrace{\Bigl(\frac{P-2}{2}+4W\Bigr)T_C}_{\text{通信残差}}
\}. 
$$

在常用近似 $T_B\!=\!2T_F$、忽略通信 $T_C$ 时，可化为气泡占比随 $W$ 单调下降的闭式：

$$
\text{BubbleRatio}\;\propto\;\frac{2P-2}{\,3PW+P-1\,},
$$

即波数翻倍，气泡近似减半；这也是文中图示“wave=2,4 气泡率依次降低”的来源。

> 直观解释：与 VPP 将单次跃迁的“时隙宽度”切细不同，Hanayo 通过增加阶段总数 $S$ 让一轮内出现 $W$ 个波；波之间彼此“接力”，把不可避免的首/尾空档摊得更细，暴露气泡随 $W$ 缩小。文中实验显示吞吐可随 $W$ 增长而上升，并对优化后 Chimera 取得最高 30.4%的加速。

#### 内存语义：

* **不复制模型**：区别于 Chimera 的双向复制（参数×2），Hanayo 不做模型副本；因此权重内存 $M_w$ 与激活内存 $M_a$ 量纲与主流同步流水相当。实际测评中，其峰值显存分布更均衡，有助于整体利用率提升。
* **激活生命周期**：仍可与梯度检查点等通用手段叠加；wave-like 通过更细的阶段推进“边算边消耗”激活，使不同设备的峰值更平滑（而非抬高量纲）。

#### 通信语义：

更多边界，更多条数：阶段数 $S$ 增大意味着跨设备边界次数增加，消息条数上升；Hanayo 依靠 action list + 预取 + 异步收发（NCCL batch\_isend\_irecv） 把大部分传输叠在下一片计算后侧，从而使“条数↑”不必然转化为“暴露时延↑”。

#### 与 1F1B / VPP 的关系

* **对 1F1B**：同属同步流水家族；Hanayo 通过 $W$ 的“波级细化”进一步**摊薄首尾空档**。
* **对 VPP**：VPP 在每卡内引入 $v$ 个虚拟阶段细化时间粒度；Hanayo 通过增加全局阶段数 $S$ 并引入 $W$ 个波来细化迭代结构。两者针对的“细化维度”不同，原则上可组合（需综合评估通信与调度复杂度）。

总结：Hanayo 用“波（$W$）”把一轮前后向拆成多段接力式推进，在不复制模型的前提下，让气泡随 $W$ 近似按 $1/W$ 缩小；配合预取和异步通信，在多种集群上取得了对 SOTA 的显著吞吐优势。



## 参考文献：
https://zhuanlan.zhihu.com/p/650744349
https://zhuanlan.zhihu.com/p/701716465
https://medium.com/@dpratishraj7991/demystifying-virtualpipeline-parallelism-in-llama-3-model-training-faf2fe7e60e5
https://blog.csdn.net/just_sort/article/details/135981391
https://blog.csdn.net/HaoBBNuanMM/article/details/134095326
https://github.com/NVIDIA/Megatron-LM
