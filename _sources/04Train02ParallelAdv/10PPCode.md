<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 10.Megatron 中 PP 实现

> Author by：高亮  

!!!!!!!!1）首先注意格式；2）注意不要大模型的列表，用自己的语言来表达和理解。3）代码的架构图，逻辑图，要去梳理这些才能深入。4）标题尽可能在 10 个字左右压缩下，不要太大长。

## 1. 模型运行入口与 PP / VPP 配置

在 Megatron-core 分布式训练框架里，通过 pretrain_gpt.py 的 main 函数调用 pretrain->get_model,get_model 函数判断 pipeline 的划分策略，其过程为：

```text
main → pretrain → setup_model_and_optimizer → get_model
```

其中`get_model` 依据启动参数决定是否启用流水线并行 p 与虚拟流水 v：

* **p ≤ 1**：整模在本卡，不做纵向切分；
* **p > 1 且 v=0**：每卡 1 段（PP）；
* **p > 1 且 v>0**：每卡 v 段（VPP，返回模型列表，训练期用 interleaved 1F1B 调度）。

首/末段用 `pre_process / post_process` 标识；VPP 额外携带 `vp_stage` 索引。

### 1.1 入口与构建触发源码

!!!!!!!!伪代码要有原理解读，其实代码可以少一点，更多的是对架构的理解和梳理

```python
# Megatron-LM/pretrain_gpt.py
if __name__ == "__main__":
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}, store=store)

# Megatron-LM/megatron/training/training.py
model, optimizer, sched = setup_model_and_optimizer(model_provider, model_type)
model = get_model(model_provider, model_type)  # ← 这里决定 PP / VPP 形态
```

### 1.2 get_model 伪代码

```pseudo
def get_model(model_provider, model_type):
  args = get_args()
  p = args.pipeline_model_parallel_size
  v = args.virtual_pipeline_model_parallel_size


  if p <= 1:
    return model_provider(pre_process=True, post_process=True)

  if not v:  # 纯 PP：每卡一段
    pre  = is_pipeline_first_stage()   # rank == 0
    post = is_pipeline_last_stage()    # rank == p-1
    return model_provider(pre_process=pre, post_process=post)

  # VPP：每卡 v 段（不相邻），返回列表，训练期按 vp_stage 轮转
  models = []
  for vp in range(v):
    pre  = is_pipeline_first_stage(vp_stage=vp)
    post = is_pipeline_last_stage(vp_stage=vp)
    m_i  = model_provider(pre_process=pre, post_process=post, vp_stage=vp)
    models.append(m_i)
  return models
```

get_model()只做两件事：判定是否 VPP，以及给当前（虚拟）stage 标好首/末段再实例化模型；返回的形态（单模型 or 模型列表）直接决定训练期选择的日程。
> 非 VPP → 返回单模型 → forward_backward_pipelining_without_interleaving（1F1B）,VPP → 返回模型列表（按 vp_stage） → forward_backward_pipelining_with_interleaving（交错轮转）。

---
!!!!!!!!大模型生成的格式内容，自己注意去掉一下，自己用工具画画图，做到真正的理解。

## 2. PP 模型实例化

让“当前 rank（或本卡的某个 `vp_stage`）只实例化自己负责的 GPT 子模型”，并保留“首段做输入、末段做输出”的语义。其流程为：

```text
main → pretrain → setup_model_and_optimizer → get_model → model_provider → GPTModel → TransformerBlock
```

- `training.get_model(...)`
   上节介绍，判断是否为 VPP，调用脚本回调 `model_provider(...)`。

- `pretrain_gpt.model_provider(pre_process, post_process, vp_stage)`
   返回 仅含本段职责 的 `GPTModel`。
   > 核心理解：PP（无论是否调用 VPP）本质上是纵向将模型切分，因此一个 stage（vp_stage）仅负责模型的部分计算，因此 PP 模型实例化本质上是给各个 stage 划分自己负责的模型部分并且实例化。后续是具体构建细节，**理解到这层即可**。

   ```python
   def model_provider(pre_process=True, post_process=True, vp_stage=None):
       return GPTModel(
           config=transformer_config,
           transformer_layer_spec=get_gpt_layer_local_spec(),  # ← 层规格
           pre_process=pre_process, post_process=post_process, vp_stage=vp_stage,
           vocab_size=..., max_sequence_length=...
       )
   ```

- `gpt_model.GPTModel(...)`
   内部创建 `TransformerBlock(...)`，把 `pre_process / post_process / vp_stage` 下传，使首段含 Embedding/输入、末段含 logits/loss，其余段只算中间层。

   ```python
   self.decoder = TransformerBlock(
       config=self.config,
       spec=get_gpt_layer_local_spec(),  # 单层如何构成由 spec 决定
       pre_process=self.pre_process, post_process=self.post_process, vp_stage=vp_stage,
   )
   ```

- `gpt_layer_specs.get_gpt_layer_local_spec()`
   返回 一层 Transformer 的构造蓝图（ModuleSpec）：包含 Self-Attention、MLP、LayerNorm、Bias-Dropout-Add 等子模块的实现（TE 或本地 kernel），供后续批量复制成“本段的多层”。

## 3. PP 获取需要执行的层数

当前 Stage（或 `vp_stage`）需要实例化的层数由`get_num_layers_to_build(config, vp_stage)` 决定，记为 N，在典型同构场景下：无 VPP 时 N ≈ L / p；有 VPP（每卡 v 段）时 N ≈ L / (p·v)，其中 L 为全局模型的总层数（Transformer 总层数）。随后在实例化时通过`get_transformer_layer_offset(config, vp_stage)` 计算全局层号起点 `offset`，将局部层号 `1..N` 映射到全局层号，保证跨 Stage/虚拟段不重叠且顺序正确（VPP 为 stride = p 的交错映射）。

### 3.1 份额计算：确定本段要建几层

```python
# megatron/core/transformer/transformer_block.py
def get_num_layers_to_build(config, vp_stage=None) -> int:
    return num_layers_to_build  # ← N

class TransformerBlockSubmodules:
    def _get_block_submodules(...):
        elif isinstance(spec, ModuleSpec) and issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage)  # ← N
            return TransformerBlockSubmodules(layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl)
```

把“单层的构造规格（`spec`）”复制 N 份，形成本段需要实例化的层列表。对应地：

```python
self.num_layers_per_pipeline_rank = len(self.layers)  # == N
```

### 3.2 全局定位：给每层标注全局层号

```python
# megatron/core/transformer/transformer_block.py
def _build_layers(self):
    def build_layer(layer_spec, layer_number):
        global_layer_number = layer_number + get_transformer_layer_offset(self.config, self.vp_stage)  # 1-based
        layer_config = (self.config.get_config_for_layer(global_layer_number)
                        if self.config.heterogeneous_block_specs else self.config)
        ...
```

 含义：用 `offset = get_transformer_layer_offset(...)` 把局部层号→全局层号。
 * 无 VPP：各 Stage 连续切片；
 * 有 VPP：同卡不同 `vp_stage` 按 stride = p 交错到全局层序列（彼此不相邻）；
 * 开启 `heterogeneous_block_specs` 时，可按 global\_layer\_number 为不同层选择不同配置/内核。

**直观例子：**

* `L=24, p=4, v=1` → `N=6`；每个 Stage 6 层、全局层号连续切分。
* `L=24, p=4, v=2` → 每个 `vp_stage` `N=3`；同卡两个 `vp_stage` 交错映射到全局层序列（步长 4）。


## 4. 执行 PP 训练

在一次 iteration 中，`train_step()` 按当前并行形态选择合适的流水线调度器并执行前向/反向：

* **无 PP**：单机单段（`forward_backward_no_pipelining`）。
* **PP，非 VPP**：每卡 1 段，1F1B 调度（`forward_backward_pipelining_without_interleaving`）。
* **PP + VPP**：每卡 $v$ 个虚拟段，Interleaved 1F1B（`forward_backward_pipelining_with_interleaving`）。

```python
# megatron/training/training.py
def train_step(...):
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,                                  # VPP: 为 [model_vp0, ..., model_vp{v-1}]
        num_microbatches=get_num_microbatches(),      # m
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
        adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
    )

# megatron/core/pipeline_parallel/schedules.py
def get_forward_backward_func():
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            return forward_backward_pipelining_with_interleaving   # PP+VPP
        else:
            return forward_backward_pipelining_without_interleaving # 纯 PP（1F1B）
    return forward_backward_no_pipelining
```

pipeline_parallel.schedules 模块作为 PP 的调度核心，为训练主循环提供一个统一入口`get_forward_backward_func()`，并且按配置返回如上所示，三选一的调度器，其调度器的内部依赖一组原子步骤，如：`forward_step`、`backward_step`、`deallocate_output_tensor`、`custom_backward` 等；并通过 pipeline_parallel.p2p_communication 完成跨 stage 的 send/recv 通信操作。

## 5. 前向过程

Megatron-Core 的前向阶段由 `forward_step` 驱动；它调用你提供的 `forward_step_func` 完成本段模型的前向，并把结果交给 `forward_step_calc_loss` 做损失/统计或收集非损失数据。官方 API 对两者的职责与参数有明确约定。

此处，会判断 PP 阶段来获取不同的数据来源，若处于首段则直接从 `data_iterator` 取 batch 作为输入。若处于非首段则会使用上游阶段通过 P2P 发送下来的 `input_tensor`。返回值均是“前向输出对象（张量或张量集合）”。

### 5.1 `forward_step` 的最小调用约定

!!!!!!!!!!格式尽可能简单

`forward_step`的核心参数是`forward_step_func`：其函数签名为：forward_step_func(data_iterator, model) -> (output_obj, reducer_fn)，其输入输出为：
- data_iterator：若首段直接取数据；非首段通常由调度层先 recv_forward 再传入。
- model：本 PP/VPP 段的子模型。
- output_obj：张量或张量集合，后续会作为 reducer_fn 的输入。
- reducer_fn：对 output_obj 做“损失归约 / 重排 / 直传”的函数，返回以下三类之一（触发不同的内部处理逻辑）：
    
    二元组 (reduced_loss, other)：框架会将 reduced_loss 再除以“全局 microbatches 数”，以保证跨卡/切分规模变化时损失数值稳定。
    三元组 (reduced_loss, num_tokens, other)：在二元组基础上，reduced_loss 还会按 num_tokens 进一步做 token 平均。
    任意对象 any_payload（如推理时需要的字典/列表/张量集合）：需在调用调度例程时设置 collect_non_loss_data=True（常与 forward_only=True 搭配），表示不计算损失，只收集前向产出。

### 5.2 `forward_step` 伪代码

```pseudo
def forward_step(forward_step_func, data_iterator, model,
                 num_microbatches, input_tensor, forward_data_store,
                 config, cp_group_size,
                 collect_non_loss_data=False,
                 checkpoint_activations_microbatch=None,
                 is_first_microbatch=False,
                 current_microbatch=None,
                 vp_stage=None, is_last_stage=True):

  # 1) 准备本段输入
  batch = next(data_iterator) if is_first_stage(config) else input_tensor

  # 2) 前向：由用户封装的 forward_step_func 执行
  output_tensor, loss_func = forward_step_func(batch, model)

  # 3) 计算/收集(损失|统计|非损失数据)
  info = forward_step_calc_loss(model, output_tensor, loss_func, config,
                                vp_stage, collect_non_loss_data,
                                num_microbatches, forward_data_store,
                                cp_group_size=cp_group_size,
                                is_last_stage=is_last_stage)

  return output_tensor, info
```

### 5.3 `forward_step_calc_loss`：三种“归约/整理”返回模式

`loss_func(output_tensor)` 允许三种返回形式，不同形式触发不同的内部缩放/汇总规则：
- 对于二元组，框架会将`reduced_loss` 再除以全局 microbatches 数，保证拆分规模变化时损失稳定。
- 对于三元组，在二元组的基础上，还会按 num_tokens 做进一步平均。
- 对于任意对象 any，需要上层在调用调度函数时设置 `collect_non_loss_data=True`（通常也会 forward_only=True）。

归约路径伪代码：

```pseudo
def forward_step_calc_loss(model, output_tensor, loss_func, config, vp_stage,
                           collect_non_loss_data, num_microbatches,
                           forward_data_store, cp_group_size=None, is_last_stage=None):

  result = loss_func(output_tensor)

  if is_tuple_len(result, 2):
    loss, other = result
    loss = loss / num_microbatches                 # 规模稳定化
    loss = maybe_reduce_across_parallel_groups(loss, cp_group_size)
    forward_data_store.append({"loss": loss, "other": other})
    return {"loss": loss, "other": other}

  elif is_tuple_len(result, 3):
    loss, ntokens, other = result
    loss = (loss / num_microbatches) / max(ntokens, eps)   # 再按 token 平均
    loss = maybe_reduce_across_parallel_groups(loss, cp_group_size)
    forward_data_store.append({"loss": loss, "ntokens": ntokens, "other": other})
    return {"loss": loss, "ntokens": ntokens, "other": other}

  else:
    assert collect_non_loss_data, "non-loss path requires collect_non_loss_data=True"
    forward_data_store.append({"data": result})
    return {"data": result}
```

## 6. 反向过程

Megatron-Core 在 PP/VPP 调度中，把一次微批的反向拆成三件事：

- 用 `deallocate_output_tensor` 在前向送出后立刻“伪释放”输出，腾出显存空间。
- 在需要回传时，用 `backward_step` 对本段做反向求梯（末段从 loss 起步、其它段先收梯度再回传）。
- 反向调用里通过 `custom_backward` 直接走 C++ autograd 引擎，以配合“伪释放”，绕开 Python `torch.autograd.backward` 的形状强校验。

### 6.1 `deallocate_output_tensor(out, deallocate_pipeline_outputs=False)` —— 前向后立刻“伪释放”激活

!!!!!标题长度

已把前向输出发给下游后，把 `out.data` 置成标量，仅保留 `out.grad_fn`，以便稍后反向；显著降低峰值显存。应当紧跟 send_forward 之后调用。为什么可以这样做呢？需要从反向过程的核心原理来理解：

反向传播依赖的是“计算图的结构 + 必要的中间缓存”而非边界输出张量本身的数值。在流水线里，一个 stage 的“边界输出”只是下游的输入；对本 stage 的反向而言，它只用来**作为反向的入口**。具体做梯度时，Autograd 会从这个入口节点（`grad_fn`）向上游回溯，逐个算子调用各自的反向函数，并使用这些算子在前向时已经保存到节点里的中间量（`saved_tensors`，如激活、索引、形状等）来计算本层参数梯度和输入梯度。也就是说：

* **需要保留的**是这个“入口节点”以及整张图的**边连接关系**（`out.grad_fn` 及其 `next_functions`）；
* **不需要保留的**是“边界输出的那份数值缓冲区”（`out.data`），因为它并不参与本 stage 的梯度公式——真正用到的中间量早就在各个算子的节点里缓存好了。

因此，在把边界输出发送给下游之后，当前 stage 可以把 `out.data` 置空（“伪释放”）以节省显存，只保留 `out.grad_fn`。等下游把对应的外部梯度（`grad_out`）传回时，我们把它“挂”到这个入口节点上，Autograd 引擎沿计算图回溯，利用各节点的 `saved_tensors` 完成参数梯度与输入梯度的计算，并不需要 `out.data`。这就是“释放大 data、仅留 grad（准确说是 `grad_fn`）仍能正常完成反向”的计算学理依据。

**伪释放伪代码**

```pseudo
# after forward on this stage
p2p.send_forward(output, is_last_stage)
deallocate_output_tensor(output)   # .data -> scalar, keep .grad_fn
```

### 6.2 `backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)`

执行逻辑可以概括为：在流水线反向阶段，梯度沿着与前向相反的方向自下游逐段回传；若当前段是末段，没有更下游可依赖，因此从本地 loss 启动反向（此时 `output_tensor_grad=None`），由 `output_tensor` 的计算图入口触发回传；若非末段，则先通过 P2P `recv_backward` 接收下游传来的对本段输出的梯度 `output_tensor_grad`，再依据链式法则在本段内部计算并累积参数梯度以及对输入张量的梯度；函数的返回值即这份“对输入的梯度”，供调度层用 `send_backward` 继续向上游传递，而当当前段是首段时，上游已不存在，因此返回 `None` 终止回传。

整个过程中，前向发送完边界输出后通常已用 `deallocate_output_tensor` 释放其数据，仅保留 `grad_fn` 作为反向入口，反向触发则通过 `custom_backward(output_tensor, output_tensor_grad)` 直接调用 C++ autograd 引擎，确保在不保留大激活数据的前提下，梯度仍能沿计算图正确回溯到本段输入与参数。

**反向过程伪代码**

```pseudo
def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    # 边界判定
    first = is_pipeline_first_stage(config)
    last  = is_pipeline_last_stage(config)

    # 1) 反向起点：末段从 loss 启动，其余段用下游传回的梯度
    grad_out = None if last else output_tensor_grad

    # 2) 触发反向（C++ autograd，引擎沿 output.grad_fn 回溯）
    custom_backward(output_tensor, grad_out)

    # 3) 返回对“输入张量”的梯度（首段无上游→返回 None）
    return None if first else input_tensor.grad
```

> 注：前向已 `send_forward(output_tensor)` 并执行 `deallocate_output_tensor(output_tensor)`；此处只需 `output_tensor` 的 `grad_fn` 与 `grad_out` 即可完成回传。

### 6.3 `custom_backward(output, grad_output)` —— 直调 C++ 引擎配合“伪释放”

为了让 `deallocate_output_tensor` 生效，必须绕开 `torch.autograd.backward` 的形状一致性检查（后者要求 `output` 与 `grad` 形状严格相同），而 C++ 引擎的 `backward` 不做该检查，能在 PP 的 send/recv / 形状调整后正常回传。

**伪代码（等价行为）**

```pseudo
def custom_backward(output, grad_output):
    # 直接调用 PyTorch C++ autograd engine：
    # torch.autograd.backward(output, grad_output, allow_unused=True)
    # 的“底层版本”，避免 Python 层 shape check。
    _cpp_autograd_engine_run_backward(output, grad_output)
```

### 6.4 反向过程中的协同

* **前向末尾**：`send_forward(output)` → `deallocate_output_tensor(output)`（显存释放，缓解压力）。
* **进入反向**：

  * **末段**：`backward_step(..., output_tensor_grad=None)` 从 loss 启动；
  * **中间段**：先通过`recv_backward()` 得到下游梯度值`output_tensor_grad`，再进行`backward_step(...)`，链式计算得到本地梯度`grad_in` 后 `send_backward(grad_in)`回传；
* **内部回传**：`backward_step` 内部用 `custom_backward` 触发 C++ 引擎回传，从而在“输出已瘦身”的前提下仍能沿计算图把梯度传播回输入。这些收发在调度里由 `P2PCommunicator` 的复合接口打包（如 `send_backward_recv_forward`/`send_forward_recv_backward`），以便通信与计算重叠。

## 7. 非交错 1F1B 调度的实现：`forward_backward_pipelining_without_interleaving(...)`

在不开启 VPP 的情况下，按**1F1B**（One-Forward-One-Backward）方式对一个全局 batch 的 `num_microbatches` 进行流水处理：先 warmup，再进入 steady“每步 1F+1B”，最后 cooldown。只有末段返回 losses 字典，其它段返回空字典；该函数也支持 `forward_only / collect_non_loss_data`。

### 7.1 核心时序

* **Warmup**：各段只做若干次**前向**并把边界输出发给下游，填满流水线；具体需要做多少个微批由 `get_pp_rank_microbatches` 计算。
* **Steady 1F1B**：每个调度步同时做“下一 microbatch 的前向 + 最早未反传 microbatch 的反向”，并用 **P2PCommunicator** 的复合接口把收/发打包以便重叠（如 `send_forward_recv_backward` / `send_backward_recv_forward`）。
* **Cooldown**：把剩余未反传的微批全部反向完成。该阶段仍按“先收下游梯度、再回传、再把输入梯度发回上游”的方向进行。

**关键实现点**

* 前向输出发给下游后，立刻 `deallocate_output_tensor(output)`：**清空 `.data`、保留 `grad_fn`**，显著降低峰值显存；稍后反向通过 `custom_backward(output, grad_out)` **直调 C++ autograd 引擎**沿图回溯。([NVIDIA Docs][1])
* 只在该“非交错”调度里提供 `adjust_tensor_shapes_fn`，可对**收/发张量形状**做一次统一调整（适配自定义分片/布局）。([NVIDIA Docs][1])

**1F1B 调度过程伪代码**

```pseudo
def forward_backward_pipelining_without_interleaving(
    *, forward_step_func, data_iterator, model,
    num_microbatches, seq_length, micro_batch_size,
    forward_only=False, collect_non_loss_data=False,
    adjust_tensor_shapes_fn=None, p2p_communicator=None, config, pg_collection=None):

    first = is_pipeline_first_stage(config)
    last  = is_pipeline_last_stage(config)

    # 0) 预备：收/发形状、warmup/剩余计数
    recv_shapes, send_shapes = infer_tensor_shapes(seq_length, micro_batch_size, config)
    if adjust_tensor_shapes_fn:
        recv_shapes, send_shapes = adjust_tensor_shapes_fn(recv_shapes, send_shapes)

    total, warmup, remaining = get_pp_rank_microbatches(num_microbatches, 1, ...)
    losses = {}

    # 1) Warmup: 只做前向并发送
    for m in range(warmup):
        inp  = (next(data_iterator) if first else
                p2p_communicator.recv_forward(recv_shapes, is_first_stage=first))
        out, info = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                 inp, forward_data_store=..., config=config,
                                 cp_group_size=..., is_last_stage=last,
                                 collect_non_loss_data=collect_non_loss_data)
        if not last: p2p_communicator.send_forward(out, is_last_stage=last)
        deallocate_output_tensor(out)

        if last: losses.update(extract_losses(info))   # 只有末段汇总

    if forward_only:
        return losses  # 评测/推理：只跑前向

    # 2) Steady 1F1B: 本步做 “下一 F + 上一 B”
    for m in range(total - warmup):
        # 2a) 前向（并与下游梯度接收重叠）
        if first:
            inp = next(data_iterator)
            out, info = forward_step(...)
            if last:
                grad_out = None
            else:
                grad_out = p2p_communicator.send_forward_recv_backward(
                    out, recv_shapes, is_last_stage=last)
            deallocate_output_tensor(out)
            if last: losses.update(extract_losses(info))
        else:
            inp = p2p_communicator.recv_forward(recv_shapes, is_first_stage=first)
            out, info = forward_step(...)
            grad_out = p2p_communicator.send_forward_recv_backward(
                out, recv_shapes, is_last_stage=last)
            deallocate_output_tensor(out)
            if last: losses.update(extract_losses(info))

        # 2b) 反向（得到对输入的梯度并发回上游）
        grad_in = backward_step(inp, out, grad_out, model_type=..., config=config)
        if not first:
            p2p_communicator.send_backward(grad_in, is_first_stage=first)

    # 3) Cooldown: 把余下未反传的微批做完反向
    for _ in range(remaining):
        grad_out = (None if last else
                    p2p_communicator.recv_backward(recv_shapes, is_last_stage=last))
        # 此处 inp/out 为该段最早未反传的那一对边界张量（由 autograd 图追溯）
        grad_in = backward_step(inp, out, grad_out, model_type=..., config=config)
        if not first:
            p2p_communicator.send_backward(grad_in, is_first_stage=first)

    return losses  # 末段：loss 字典；其它段：{}
```

## 8. 交错 1F1B 调度的实现：`forward_backward_pipelining_with_interleaving(...)`

在开启 VPP 时，将每个物理 stage 的层进一步切成 K 个 model chunks，微批在 (stage × chunk) 上交错流动，以进一步压缩 pipeline bubble。该调度要求把 `model` 与 `data_iterator` 作为列表传入（每个 chunk 对应一个元素），并在末段返回 `losses` 字典、其他段返回空字典。

### 8.1 核心时序（交错 1F1B）

* **Warmup**：按调度表，不同 stage 在不同 chunk 上做前向，并将边界输出发送给下游。
* **Steady**：每个时间隙执行一个三元组 `(virtual_microbatch_id, microbatch_id, model_chunk_id)`；同一步内把本步 chunk 的前向发送与接收更早微批的梯度打包（`send_forward_recv_backward`），随后对相应输入做反向并把 `grad_in` 发回上游。

> steady 阶段的三元组 (vmb, mb, chunk_id) ：vmb 是“在本机这条虚拟流水线上的第几个在飞槽位，mb 表示“要处理的那一个全局微批编号，chunk_id 表示“本 stage 内用哪一块模型分片（chunk）来跑这一步，其中在飞指的是：指某个微批已经在这条流水线上完成了前向，但还没完成对应的反向，因此仍占着这条虚拟流水线的一席“槽位”。因此 steady 阶段利用三元组回答了这一步到底在处理谁、在哪条虚拟线、用哪块模型的问题。

* **Cooldown**：将尚未回传完的微批逐一做反向。
  通信均由 `P2PCommunicator` 的复合接口完成，以实现通信×计算重叠；前向发送后立即 `deallocate_output_tensor` 只做显存优化、不改变时序。

### 8.2 调度表与执行次序

函数内部先构造交错的**调度表**：

* `get_schedule_table(...)` 产出每个时隙要执行的三元组 **`(vmb, mb, chunk_id)`**；
* `convert_schedule_table_to_order(...)` 将表格转为“可执行序列”。

> 直观理解：交错时序让**同一 stage**在 **chunk A / chunk B / …** 间来回切换地执行“F/B”，从而减少等待，提升吞吐。


**交错 1F1B 调度伪代码（与非交错版对齐的写法）**

```pseudo
def forward_backward_pipelining_with_interleaving(
    *, forward_step_func, data_iterator=[it0..itK-1], model=[m0..mK-1],
    num_microbatches, seq_length, micro_batch_size,
    forward_only=False, collect_non_loss_data=False,
    p2p_communicator=None, config, pg_collection=None):

    first = is_pipeline_first_stage(config)
    last  = is_pipeline_last_stage(config)

    # 0) 预备：按 chunk 推导收/发形状；生成交错时序
    shapes = [infer_tensor_shapes_for_chunk(c, seq_length, micro_batch_size, config)
              for c in range(K)]
    order  = convert_schedule_table_to_order(get_schedule_table(...))  # 列出一串 (vmb, mb, chunk_id)

    losses = {}

    # 1) 逐时隙执行（交错：每步挑一个 chunk）
    for (vmb, mb, cid) in order:
        mdl, it, shp = model[cid], data_iterator[cid], shapes[cid]

        # 1a) 前向（chunk=cid）
        inp = (next(it) if first else p2p_communicator.recv_forward(shp.recv, is_first_stage=first))
        out, info = forward_step(forward_step_func, it, mdl, num_microbatches,
                                 inp, forward_data_store=..., config=config,
                                 cp_group_size=..., is_last_stage=last,
                                 collect_non_loss_data=collect_non_loss_data)

        # 1b) 复合通信 + 显存优化
        grad_out = (None if last else
                    p2p_communicator.send_forward_recv_backward(out, shp.recv, is_last_stage=last))
        deallocate_output_tensor(out)

        if forward_only:
            if last: losses.update(extract_losses(info))
            continue

        # 1c) 反向（与该 vmb 对应的最早未回传项）
        grad_in = backward_step(inp, out, grad_out, model_type=..., config=config)
        if not first:
            p2p_communicator.send_backward(grad_in, is_first_stage=first)

        if last:
            losses.update(extract_losses(info))

    return losses if last else {}
```

### 8.3 与非交错 1F1B 的差异速览

* **相同**：都有 warmup→steady→cooldown；只在末段汇总 `losses`；使用复合通信重叠 FWD/BWD；前向发送后立即“瘦身”输出、反向用 `custom_backward` 直调 C++ 引擎。
* **不同**：交错版每步还要选择 chunk 并遵循 `get_schedule_table/convert_schedule_table_to_order` 的三元组时序，对每个 chunk 的收/发形状分别处理，并满足 VPP 的连续微批窗口/整除性等额外约束。

## 9. 无流水线：`forward_backward_no_pipelining(...)`

当 PP size = 1（即没有跨卡流水线）或仅做单段验证/推理时，训练循环会走这条路径：不做任何 P2P 发送/接收，整模在本 rank 内完成 FWD/BWD/优化相关的梯度累积。末段/非末段的区分自然消失，`losses` 直接在本段汇总返回；若 `forward_only=True` 或 `collect_non_loss_data=True`，仅做前向与数据收集。

**关键点**

* **无 P2P 通信**：没有 `send_forward/recv_forward`、`send_backward/recv_backward`。
* **常规 Autograd 路径**：仍使用你提供的 `forward_step_func` 和（可选的）`custom_backward`；是否使用 `deallocate_output_tensor` 取决于你是否希望在单段下也做显存优化（通常可省略）。
* **梯度累积**：对 `num_microbatches` 按常规在本地累积，随后进入优化器步（若配置了分布式优化器/TP/DP，这些在各自维度单独处理，与“无流水线”无冲突）。
* **推理/验证**：`forward_only=True` 时，不触发反向与优化器步。

**无流水线并行伪代码**

```pseudo
def forward_backward_no_pipelining(
    *, forward_step_func, data_iterator, model,
    num_microbatches, forward_only=False,
    collect_non_loss_data=False, config, pg_collection=None):

    losses = {}
    for mb in range(num_microbatches):
        # 前向
        inp = next(data_iterator)
        out, info = forward_step(forward_step_func, data_iterator, model,
                                 num_microbatches, inp,
                                 forward_data_store=..., config=config,
                                 cp_group_size=..., is_last_stage=True,
                                 collect_non_loss_data=collect_non_loss_data)

        if forward_only:
            continue  # 仅收集前向产出（评测/推理）

        # 反向（无 P2P，直接本地回传）
        custom_backward(out, grad_out=None)   # 从本地 loss 启动
        # （参数梯度已在本地累积）

        # 末段/唯一段：汇总 loss
        losses.update(extract_losses(info))

    return losses  # 单段直接返回（若 forward_only=True 可能为空/只含指标）
```

`forward_backward_no_pipelining` 是“单段”场景下的统一执行路径：不做跨段通信，在本 rank 内完成前向、（可选）反向与梯度累积；`losses` 直接在本地汇总，适用于 PP=1 或仅推理/验证的运行模式。

## 参考与引用

- [1] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/README.md
- [2] https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/pipeline_parallel.html#