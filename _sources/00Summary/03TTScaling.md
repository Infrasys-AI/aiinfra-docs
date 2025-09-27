<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 03. Inference Time Scaling

Author by：侯宇博

推理端的 scaling law 更关注 推理延迟、显存、计算复杂度随模型规模和上下文长度变化的规律。其中Inference/test time scaling，其核心思想是在模型推理（Inference）阶段，通过投入更多计算资源以生成更多的输出 token，进而增强模型的逻辑推理（Reasoning）能力。

该方法的基本原理在于，生成单个 token 的过程（即一次模型前向传播）所包含的计算量是固定的。对于需要多步逻辑推演的复杂问题，模型无法在单次计算中完成求解。因此，必须通过生成一系列包含中间步骤的文本（即更多的 token），来逐步展开其“思考”过程，从而解决问题。

那么有哪些方法可以帮助模型产生中间推理步骤呢？

## 优化推理输入：思维链提示

思维链提示（Chain-of-Thought Prompting）通过在少样本示例中展示一系列中间推理步骤，而不仅仅是最终答案，来引导大型语言模型在解决问题时，也自主地生成类似的“思考过程”，从而释放其内在的复杂推理潜力。

![COT](../images/00Summary/02TTScaling01.png)

然而，思维链提示需要为特定任务精心设计推理示例，这限制了其通用性与易用性。一个自然而然的问题是：能否让模型在没有任何范例的情况下，仅根据问题本身就自动生成思维链？

## 优化推理输出：后训练中的Inference Time Scaling

[Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)

## Agent中的Inference Time Scaling

[Deep researcher with test-time diffusion](https://research.google/blog/deep-researcher-with-test-time-diffusion/)

## 参考资料

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
- [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)
- [Deep researcher with test-time diffusion](https://research.google/blog/deep-researcher-with-test-time-diffusion/)
