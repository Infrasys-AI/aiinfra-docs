<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 大模型后训练与强化学习

大模型后训练与RL正从“辅助对齐手段”跃升为“塑造模型行为的核心引擎”。传统SFT已难以满足复杂人类偏好，以RLHF为代表的后训练技术，通过奖励建模与策略优化，使模型输出更安全、有用、符合价值观。DPO、KTO、IPO等无需显式奖励模型的算法迅速崛起，大幅降低训练复杂度，提升稳定性与可复现性。同时，后训练正走向多目标协同：兼顾事实性、创造性、安全性与风格控制，并与在线学习、持续对齐、模型自我进化结合。RL 不再局限于PPO，CPO、Reinforce++、GRPO等新算法在样本效率与训练鲁棒性上持续突破。后训练，已成为大模型从“聪明”走向“可控、可信、可协作”的关键跃迁。

## 详细内容

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 | 状态 |
|:--- |:---- |:-------------------- |:---:|
|  |  | [PPT](), [文章](), [视频]() |  |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| RL 实践 :computer: | CODE 01: 经典 InstructGPT 复现 | [Markdown](./Code01InstructGPT.md), [Jupyter](./Code01InstructGPT.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train04PostTrainRL/Code01InstructGPT.html) | :white_check_mark: |
| RL 实践 :computer: | CODE 02: DPO 与 PPO 在 LLM 对比 | [Markdown](./Code02DPOPPO.md), [Jupyter](./Code02DPOPPO.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train04PostTrainRL/Code02DPOPPO.html) | :white_check_mark: |
| RL 实践 :computer: | CODE 03: LLM + GRPO 实践  | [Markdown](./Code03GRPO.md), [Jupyter](./Code03GRPO.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train04PostTrainRL/Code03GRPO.html) | :white_check_mark: |

## 备注

系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您参与到这个开源课程中，B 站给 ZOMI 留言哦！
>

```{toctree}
:maxdepth: 1

Code01InstructGPT
Code02DPOPPO
Code03GRPO
DAPO
GRPO
GSPO
PPO_DPO
RLHF
RLVR
VAPO
```