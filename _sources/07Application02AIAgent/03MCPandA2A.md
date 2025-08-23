<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# MCP 与 A2A 原理

## MCP 原理

2023年 OpenAI 首次在大模型引入 [Function Call](https://platform.openai.com/docs/guides/function-calling) 的概念，通过结构化的输出，使得大模型不再局限于做一个聊天的工具，而是可以真实的调用外部的API，让大模型具备了自主和外部系统交互、使用外部工具的能力，大大拓展了大模型能力的边界。

2024 年底，Anthropic 推出了 开源协议 Model Context Protocol（[MCP](https://modelcontextprotocol.io/introduction)）也即模型上下文协议，目标是统一应用程序和 AI 模型之间交换上下文信息的方式。使开发者能够以一致的方式将各种数据源、工具和功能连接到 AI，它是一个中间协议层，就像 USB-C 让不同设备能够通过相同的接口连接一样。

从提出以来，该协议得到了开源社区和商业组织的广泛关注和积极响应。并基于此协议提供了大量的 [MCP Server](https://github.com/modelcontextprotocol/servers)。某种程度上，MCP已经成为了 AI 工具调用的行业标准。


### MCP 基本介绍

### 从 Function Call 到 MCP，为什么需要 MCP

### MCP Demo 演示

## MCP 核心原理

### MCP 设计理念

### MCP 架构

### 一次 MCP 的完整调用之旅

## 总结与思考

### MCP 的局限性

### MCP 未来展望

## A2A 原理
