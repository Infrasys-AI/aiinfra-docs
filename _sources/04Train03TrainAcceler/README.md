<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 大模型训练加速

系统讲解大模型训练加速的核心算法与技术，涵盖计算优化（Flash Attention 的演进、DS GEMM 稀疏计算与 MTP 并行）、序列优化（Ulysses 多流调度与 Ring Attention 长序列处理）、内存优化（MLA 注意力机制与梯度检查点）、混合精度训练（FP8 表示与稳定性）以及通算融合（DeepSpeed-Domini 与 FLUX 的通信计算重叠）等关键方向，旨在通过原理剖析与实现细节全面了解如何提升大模型训练效率。

## 详细内容

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 | 状态 |
|:--- |:---- |:-------------------- |:---:|
| 大模型训练加速 |   | [PPT](), [文章](), [视频]() | |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| 并行 实践 :computer: | CODE 01: Flash Attention 实现 | [Markdown](./Code01FlashAtten.md), [Jupyter](./Code01FlashAtten.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train03TrainAcceler/Code01FlashAtten.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 02: 梯度检查点内存优化 | [Markdown](./Code02GradCheck.md), [Jupyter](./Code02GradCheck.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train03TrainAcceler/Code02GradCheck.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 03: FP8 混合精度训练  | [Markdown](./Code03FP8.md), [Jupyter](./Code03FP8.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train03TrainAcceler/Code03FP8.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 04: Ring Attention 实践 | [Markdown](./Code04RingAttn.md), [Jupyter](./Code04RingAttn.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train03TrainAcceler/Code04RingAttn.html) | :white_check_mark: |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

```{toctree}
:maxdepth: 1

01Introduction
02FlashAttn
03DSGEMM
04MTP
05Ulysses
06RingAttn
07MLA
08Checkpoint
09HybridPrecision
10Domino
11FLUX
Code01FlashAtten
Code02GradCheck
Code03FP8
Code04RingAttn
```