<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 大模型验证评估

大模型评估正从“单一指标打分”迈向“多维动态对齐”的新阶段。随着模型能力跃升，传统自动指标（如 BLEU、ROUGE）已难以反映真实语义与人类偏好，业界正转向以 GPT-4、Claude 等强模型作为裁判的自动化评估（如 AlpacaEval、MT-Bench），并结合多维度人工评测（有用性、安全性、事实性、风格一致性）。同时，评估范式从静态测试集走向动态对抗（如 HELM、Big-Bench Hard）、领域专项（医疗、法律、代码）和长上下文压力测试。近年来，评估更强调“对齐人类意图”与“价值观校准”，引入偏好一致性、幻觉检测、可解释性追踪等新维度。大模型评估，已不仅是性能度量，更是模型可信、可控、可落地的核心保障体系。

## 详细内容

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 | 状态 |
|:--- |:---- |:-------------------- |:---:|
|  |  | [PPT](), [文章](), [视频]() |  |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| EVA 实践 :computer: | CODE 01: OpenCompass 评估实践 | [Markdown](./Code01OpenCompass.md), [Jupyter](./Code01OpenCompass.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train06VerifValid/Code01OpenCompass.html) | :white_check_mark: |

## 备注

系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您参与到这个开源课程中，B 站给 ZOMI 留言哦！
>

```{toctree}
:maxdepth: 1

01Introduction
Code01OpenCompass
```