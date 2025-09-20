<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 05.Selective 微调算法

> Author by: 李亚鹏

!!!!!!!!!
1）开篇介绍，这是一篇技术文章，不要一上来就直接算法原理；2）算法流程图，架构图；3）用心点，一看就给人的感觉太敷衍了。

## XXXX

### U-Diff-Pruning

**U-Diff-Pruning**（无结构差分剪枝）：对于每个新任务，学习一个非常稀疏的“差分向量”（diff vector） $δ_τ$ ，这个向量被加到原始参数上，从而使模型适应新任务。

$$
\theta_\tau=\theta+\delta_\tau
$$

为了鼓励稀疏性，最直接的方法是惩罚中非零元素的数量。这个数量由 L0 范数（ $||δ_τ ||_0 $ ）来度量。因此，优化目标变成了：

$$
\begin{aligned}\min_{\delta_\tau}&L(\mathcal{D}_\tau,f_\tau,\boldsymbol{\theta}+\boldsymbol{\delta}_\tau)+\lambda R(\boldsymbol{\theta}+\boldsymbol{\delta}_\tau)\end{aligned}
$$

$$
R(\boldsymbol{\theta}+\boldsymbol{\delta}_\tau)=\|\boldsymbol{\delta}_\tau\|_0=\sum_{i=1}^d\mathbb{1}\{\boldsymbol{\delta}_{\tau,i}\neq0\}
$$

然而，L0 范数是不可导的，所以不能直接用梯度下降法来优化，这是剪枝领域的经典难题。

为了解决 L0 范数不可导的问题，进行 L0 范数的可微近似。 

分解差分向量：将 $δ_τ$ 分解为一个掩码向量(mask vector)$z_τ$ 和一个潜在权重向量 (potential weight vector) $w_τ$ 的逐元素乘积。

$$
\boldsymbol{\delta}_\tau=\mathbf{z}_\tau\odot\mathbf{W}_\tau
$$

其中， $w_τ$ 表示一个稠密的、可学习的原始参数向量； $z_τ$ 表示一个理想情况下由 0 和 1 组成的二元掩码，决定了 $w_τ$ 中的哪些元素被“激活”。若 $z_(τ,i)=0 $，那么 $δ_(τ,i)=0$，即实现了剪枝。

此时的二元掩码 $z_τ$ 仍是不可微分的，为了对其进行松弛，进行 Hard-Concrete distribution 操作：

先为每个参数引入一个可学习的“概率”参数 $α_τ$ ，$α_τ$ 控制了第 $i$ 个参数是 0 还是 1 的倾向。

从一个标准的均匀分布中采样一个随机数 $u$：

$$
\mathbf{u}\sim U(\mathbf{0},\mathbf{1})
$$

将 $u$ 转化为一个服从逻辑分布的连续随机变量 $s_τ$，其值受到随机噪声 $u$ 和可学习参数 $α_τ$ 的共同影响。

$$
\mathbf{s}_\tau=\sigma\left(\log\mathbf{u}-\log(1-\mathbf{u})+\boldsymbol{\alpha}_\tau\right)
$$

$s_τ$ 的分布总是在 (0, 1) 开区间内，永远不会正好等于 0 或 1。这对于实现真正的剪枝（即参数正好为 0）是不利的。

因此， Hard-Concrete 先将 $s_τ$ 拉伸到(l, r)区间，其中 $l<0$，$r>1$。再进行裁剪，将[0, 1]区间外的值置为确定的 0/1。

$$
\bar{\mathbf{s}}_\tau=\mathbf{s}_\tau\times(r-l)+l
$$

$$
\mathbf{z}_\tau=\min(\mathbf{1},\max(\mathbf{0},\mathbf{\bar{s}}_\tau))
$$

此时，正则项可微，目标函数修正为：

$$
\min_{\boldsymbol{\alpha}_\tau,\mathbf{w}_\tau}\mathbb{E}_{\mathbf{u}\sim U[\mathbf{0},\mathbf{1}]}\left[L(\mathcal{D}_\tau,f_\tau,\boldsymbol{\theta}+\mathbf{z}_\tau\odot\mathbf{w}_\tau)\right]+\lambda\sum_{i=1}^d\sigma\left(\boldsymbol{\alpha}_{\tau,i}-\log\frac{-l}{r}\right)
$$

使用修正的目标函数，即可实现学习稀疏的差分向量的目标。

### SARA

**SARA**，Sparse Low-Rank Adaptation：受到剪枝的启发，在评估参数重要性后，不将影响低的参数剪掉，而是用利用这些暂时不重要的参数进行下游任务的训练。换句话说，就是优化稀疏权重矩阵（不重要参数矩阵）来学习特定任务的知识。

!!!!!!!!
markdown 格式，打不开
<img src="images\06Selective 算法：Unstructural Masking01.png" style="zoom: 25%;" />

模型的大多数参数值都在 0 的附近。设定一个阈值，权重绝对值低于这个阈值的参数被置为 0。实验证明，5e-4 到 1e-3 的阈值下，置 0 操作对模型原始能力的影响微乎其微，也就是说，绝对值低于阈值的参数是无效（不重要）参数。

实验证明，由于训练过程的随机性而导致的初始无效参数，大部分随着训练时间的推移变得有效，可以利用这些暂时无效的参数来微调预训练的模型。

对潜在有效参数（初始无效参数）进行微调：

$$
\begin{aligned}\nabla P_M&=\nabla P\odot M+\mathbf{0}\odot(1-M)\\P_{new}&=P-\lambda\cdot\nabla P_M\end{aligned}
$$

其中 $M$ 是掩码矩阵，$P_M$ 为潜在有效参数矩阵。梯度、更新只涉及 $P_M$。

SARA 微调方法很好的利用了初始无效参数，将这部分看似“无效”的参数重新利用起来，使其在下游任务微调中发挥作用。

## Unstructural Masking

### S-Diff-Pruning

!!!!!!!!
一个算法没有图？流程图架构图

**S-Diff-Pruning**（Structural Diff-Pruning ）：基于上一小节的基础方法 U-Diff-Pruning（ Unstructural Diff-Pruning ）拓展而来。

U-Diff-Pruning 是独立地决定每个参数是否被剪枝。但模型中的参数通常是有结构的（例如，一个权重矩阵或一个偏置向量）。结构化剪枝的思想是，要剪就一起剪掉整个结构。

这鼓励模型将整个参数块（如某个注意力头的权重矩阵）完整保留或完整剪枝，实验证明这种结构化的方法性能更好。

S-Diff-Pruning 将参数根据结构进行分组，例如一个层的所有权重为一组。除了已有的掩码 $z_τ$，额外引入一个组内共享掩码 $z_τ^g$。

一个参数最终被保留，当且仅当它自身的掩码 $z_(τ,i)$ 和它所属的组的共享掩码 $z_τ^g$ 都为“开启”状态。正则项修正如下：

$$
\mathbb{E}\left[\left\|\boldsymbol{\delta}_\tau\right\|_0\right]=\sum_{j=1}^G\sum_{i\in g(j)}\mathbb{E}\left[\mathbb{1}\{\mathbf{z}_{\tau,i}\cdot\mathbf{z}_\tau^g>0\}\right]
$$

## SPT

Adapter、LoRA 等方法通常在所有下游任务中都对模型相同的位置（例如，所有 Transformer 块的 self-attention 部分）插入可训练参数，而忽略了不同任务可能需要调整模型不同部分这一事实。

**SPT**，Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning：不应对所有任务都调整相同的参数，而应该根据具体任务，自适应地选择在哪里以及如何分配有限的可训练参数预算。

SPT 的核心思想是：首先识别出对于特定下游任务最重要的（即最“敏感”的）参数，然后根据一个固定的参数预算，智能地分配微调资源。

理想情况下，一个参数 $w$ 的敏感度 $s$ 可以定义为：只微调这一个参数能给任务损失(loss)带来多大的降低。但为模型中所有参数进行这种完整的计算，成本是巨大的。

为此，SPT 使用一阶泰勒展开来近似 loss 的变化，即参数 $w_n$ 的敏感度 $s_n=g_n△w_n$，其中 $g_n$ 为梯度。

进一步的，将参数更新 $△w_n$ 近似为单步梯度下降的结果，即 $△w_n=ε g_n$，其中ε为学习率。

由此，$s_n=g_n^2$ 。由于对于所有参数来说学习率一致，因此在比较中忽略学习率。

 因此，一个参数的敏感度可以直接用它在任务数据上的梯度的平方来衡量。梯度越大，说明该参数对当前任务的损失影响越大，调整它可能带来的收益也越大，因此它就越“敏感”。

实际操作中，选取部分训练样本，进行参数梯度的计算，从而近似得到所有参数的敏感度。

对于可训练参数预算的分配，SPT 综合考虑了非结构化与结构化微调策略。

具体来说，首先依据敏感度得分 $S$，选出 $Top-τ$ 的高敏感参数。

结构化策略：以矩阵为微调单位，若矩阵 W 中的高敏感参数数量大于阈值 $σ_{opt}$，则对整个矩阵进行微调，具体方法为使用 LoRA 或 Adapter，引入小的、新的可训练模块。（如 $W_{up}$ 和 $W_{down}$）

非结构化策略：若矩阵 W 中的高敏感参数数量小于阈值 $σ_opt$，只更新掩码 $M$ 标记的高敏感参数，其中 $g_W$ 为 $W$ 的梯度。

$$
\boldsymbol{W}^{\prime}=\left\{\begin{array}{ll}\boldsymbol{W}+\boldsymbol{W}_\mathrm{down}\boldsymbol{W}_\mathrm{up}&\quad\mathrm{if}\quad\sum_{j=0}^{d_\mathrm{in}\times d_\mathrm{out}}\boldsymbol{M}^j\geq\sigma_\mathrm{opt}\\\boldsymbol{W}-\epsilon\boldsymbol{g}_W\odot\boldsymbol{M}&\quad\mathrm{otherwise}\end{array}\right.
$$

SPT 自适应地结合了结构化和非结构化调优粒度，对应实现了更强的表示能力和更高的灵活性，并将训练参数控制在了预算内。

## 总结与思考

!!!!!!!!!!

## 参考与引用

- Diff-Pruning：Parameter-Efficient Transfer Learning with Diff Pruning， https://arxiv.org/abs/2012.07463
- SPT：Sensitivity-aware visual parameter-efficient fine-tuning，https://arxiv.org/abs/2303.08566
- Diff-Pruning：Parameter-Efficient Transfer Learning with Diff Pruning， https://arxiv.org/abs/2012.07463
- SARA：Sparse Low-rank Adaptation of Pre-trained Language Models， https://arxiv.org/abs/2311.11696