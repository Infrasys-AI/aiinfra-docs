对于主流大模型例如DeepSeek-R1-671B，Qwen3-32B，我们知道B 表示 Billion，表示他们的参数量。那么给定对应的模型参数和结构，我们怎么计算模型的总参数呢？下面以Qwen3-8B和Qwen3-30B-A3B为例来深入讲解Dense模型和MOE模型总参的计算。


![alt text](./../images/04Train03TrainAcceler/image.png)

![alt text](./../images/04Train03TrainAcceler/image-2.png)




### Dense 计算总参
下面是一个Dense模型 Qwen3-8B。

``` json

Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 4096)
    (layers): ModuleList(
      (0-35): 36 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=4096, out_features=12288, bias=False)
          (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
          (down_proj): Linear(in_features=12288, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((4096,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((4096,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((4096,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)

```


从上图可以看到，大模型整个可以分为三个部分，embed_tokens 和 layers ，lm_head三个部分。

其中 embed_tokens 和 lm_head 是相似的，分别是将Token_id 转为embedding，以及将embedding转为 Token_id，所以他们有时候是一样的。对于小模型来说，比如Qwen3-0.6B，Qwen3-1.7B，Qwen3-4B，在表中可以看到Tie Embedding 为 YES，表示 embed_tokens 部分和 lm_head 是共用的。

layers 层数整个大模型的核心，由N层完全一样的结构堆叠起来。每一层又可以分为self_attn 和 mlp两个部分。

self_attn可以是MHA，GQA等，对应的参数都是Q，K，V，O四个矩阵。Qwen3使用了GQA，对于8B尺寸来说Q-Heads为32，K-V-Heads为8。故K，V矩阵 为 d * d * 1/ 4 ，Q，O矩阵依然是 d * d，其中d 为隐藏层大小，这里为4096。Qwen3系列在self_attn还使用了q_norm和k_norm，对应的参数较小，可以忽略不计。该部分总计为 2.5 * d * d。

mlp部分参数部分为gate，up和down三个矩阵，三矩阵完全相同，故总参数大小为 3 * d * ffn，其中d是隐藏层大小，ffn是mlp层升维大小，一般是3-4d，qwen3这里是3d。

在每一层之间还有两次Norm，该部分参数较小，可以忽略不计。

对于Qwen3-8B来说，总参数为 embed_tokens + 36 Layers + lm_head。

其中 embed_tokens 和 lm_head 均为  V * d，V是单词表大小，d是隐藏层大小。

每一层中，self_attn 为 2.5 * d * d，mlp部分为 9 * d * d，总计为 11.5 d。

故总参数为 2 *  V * d + 11.5 * d * d * 36 = 8190427136，故为8B

请大家根据这种方式，计算下Qwen3-32B的总参。


### MOE计算总参

下面是一个MOE模型 Qwen3-30B-A3B

``` json
Qwen3MoeForCausalLM(
  (model): Qwen3MoeModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-47): 48 x Qwen3MoeDecoderLayer(
        (self_attn): Qwen3MoeAttention(
          (q_proj): Linear(in_features=2048, out_features=4096, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
          (q_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MoeSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=128, bias=False)
          (experts): ModuleList(
            (0-127): 128 x Qwen3MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=768, bias=False)
              (up_proj): Linear(in_features=2048, out_features=768, bias=False)
              (down_proj): Linear(in_features=768, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen3MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)

``` 

对于MOE模型的分析，和Dense模型完全一样，也是分为embed_tokens 和 layers ，lm_head三个部分。其中embed_tokens 和 lm_head 完全一样，差别在于layers。

对于每一层中的mlp和self_attn，self_attn和dense是完全一样的，根据表中计算即可。

对于mlp部分来说，他这里相比起Dense模型，多了一个gate，这个gate就负责将每一个token分发对对应的专家。接下来显示有128个专家，也就是说128个这样的ffn，ffn的结构和之前是完全一样的，也是gate，up，down三个部分是完全一致的。换句话来说，MOE相比起Dense模型，他的核心就是将一个大的FFN拆分成了多个小的FFN，并且增加了gate路由机制来选择每次处理的专家ID。这一部分的总参数量为 gate + 128 * ffn。


对于Qwen3-30B-A3B来说，总参数为 embed_tokens + 48 Layers + lm_head。

其中 embed_tokens 和 lm_head 均为  V * d，V是单词表大小，d是隐藏层大小。 V = 151936， d = 2048 

每一层中，self_attn 为 4.5 * d * d，mlp部分为 d * experts + experts * d * ffn * 3 ，总计为4.5 * d * d + experts * d * ffn * 3。 d = 2048 ，experts = 128， ffn  = 768。

故总参数为 2 *  V * d + 48 * (4.5 * d * d + experts * d * ffn * 3) = 30519328768，故为30B

对于激活值来说，2 *  V * d + 48 * (4.5 * d * d + top_k * d * ffn * 3) = 3340238848，故为3B，其中top_k = 8 。

请大家根据这种方式，计算下DeepSeek-V3的总参数




