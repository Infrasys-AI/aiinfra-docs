<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 大模型微调 SFT

大模型微调正从“全参训练”的粗放模式，快速演进为“高效、可控、任务自适应”的精细化工程体系。2024 年以来，以 LoRA 为代表的参数高效微调（PEFT）已成工业标配，而 QLoRA、DoRA、PiSSA 等新方法进一步突破显存与表达力瓶颈，使 70B 级模型可在消费级 GPU 上微调。数据层面，合成指令（如 Evol-Instruct、Self-Rewarding）与课程学习策略大幅提升样本质量与训练稳定性。训练策略上，多阶段渐进微调、损失函数定制、梯度裁剪优化等技巧显著提升收敛效率与泛化能力。更重要的是，微调不再孤立，它正与 RLHF、DPO、模型编辑等对齐技术深度融合，构建“训练-对齐-评估”闭环。微调，成为释放大模型垂直领域的核心钥匙。

## 详细内容

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 | 状态 |
|:--- |:---- |:-------------------- |:---:|
|  |  | [PPT](), [文章](), [视频]() |  |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| SFT 实践 :computer: | CODE 01: Qwen3-4B 模型微调 | [Markdown](./Code01Qwen3SFT.md), [Jupyter](./Code01Qwen3SFT.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train05FineTune/Code01Qwen3SFT.html) | :white_check_mark: |
| SFT 实践 :computer: | CODE 02: LoRA 微调 SD | [Markdown](./Code02SDLoRA.md), [Jupyter](./Code02SDLoRA.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train05FineTune/Code02SDLoRA.html) | :white_check_mark: |

## 备注

系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您参与到这个开源课程中，B 站给 ZOMI 留言哦！
>

```{toctree}
:maxdepth: 1

01Introduction
05LoRA
06Selective
Code01Qwen3SFT
Code02SDLoRA
```