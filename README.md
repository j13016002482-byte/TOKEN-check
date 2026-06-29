# AstrBot Token 统计插件

<div align="center">

![AstrBot](https://img.shields.io/badge/AstrBot-v4.x+-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**一个功能强大的 AstrBot Token 使用统计插件，提供精确的 Token 消耗追踪和性能分析。**

</div>

## ✨ 功能特性

| 功能 | 说明 | 需要补丁 |
|------|------|---------|
| 📊 **精确 Token 统计** | 输入/输出/总计 Token 数 | ❌ |
| 📦 **缓存命中统计** | 显示缓存 Token 和新增处理 Token | ✅ |
| 🧠 **深度思考统计** | 显示推理 Token（阈值 >100 过滤噪音） | ❌ |
| 🔧 **工具调用统计** | 显示工具调用产生的 Token | ❌ |
| 🚀 **性能指标** | 解码速度、Prefill 速度 | ✅ (TTFT) |
| ⏱️ **TTFT 精确读取** | 首字时间（从 agent_stats 读取） | ✅ |
| 📈 **上下文分析** | 最大上下文长度、使用率 | ✅ |
| 📊 **累计统计** | 当前会话的累计 Token 消耗 | ✅ |
| 🎯 **多提供商支持** | 兼容所有 AstrBot 支持的 LLM 提供商 | ❌ |

## 📸 输出示例

```
📊 Token 统计信息
━━━━━━━━━━━━━━━━━━
🤖 模型: Qwen3.6-35B-A3B-UD-Q6_K.gguf
📝 输入 tokens: 50,785
   ├─ 缓存命中: 20,985
   └─ 新增处理: 29,800
💬 输出 tokens: 925
📊 总 tokens: 51,710
📈 上下文使用率: 40.4%
📏 最大上下文: 128,000 tokens
━━━━━━━━━━━━━━━━━━
⏱️ 响应时间: 58.4s
   ├─ Prefill (TTFT): 31.2s
   └─ Decode: 27.2s
🚀 解码速度: 34.0 tokens/s
⚡ Prefill速度: 955.1 tokens/s
━━━━━━━━━━━━━━━━━━
📈 累计统计（当前会话）:
📝 总输入: 152,355 | 💬 总输出: 2,775 | 📊 总计: 155,130
━━━━━━━━━━━━━━━━━━
```

## 🚀 安装方式

### 方式一：手动安装（推荐）

1. 下载 `main.py` 文件
2. 复制到 AstrBot 插件目录：
   ```bash
   mkdir -p ~/.astrbot/plugins/astrbot_plugin_token_stats
   cp main.py ~/.astrbot/plugins/astrbot_plugin_token_stats/
   ```
3. 重启 AstrBot

### 方式二：应用内核补丁（增强功能）

> ⚠️ **可选操作** - 不应用补丁插件仍可正常工作，但会缺少以下功能：
> - TTFT（首字时间）
> - Prefill 速度
> - 与 WebChat 一致的累计 Token 统计
> - 正确的提供商上下文长度读取

```bash
# 进入 AstrBot 安装目录
cd /path/to/AstrBot

# 应用补丁
patch -p1 < /path/to/patches/tool_loop_agent_runner.patch

# 重启 AstrBot
```

## 📋 命令列表

| 命令 | 说明 |
|------|------|
| `/token_stats` | 显示最近一次统计 |
| `/token_stats history` | 显示当前会话历史摘要 |
| `/token_stats avg` | 显示最近 10 次平均值 |
| `/token_stats clear` | 清空历史记录 |
| `/token_config` | 查看/修改配置 |

## 🔄 兼容性

- **AstrBot 版本**: v4.x+
- **Python 版本**: 3.10+
- **支持的提供商**: 所有 AstrBot 支持的 LLM 提供商

## 📄 许可证

MIT License
