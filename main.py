import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.core.provider.entities import TokenUsage
from astrbot.api.star import Context, Star, StarTools
from astrbot.api import logger
import astrbot.api.message_components as Comp


class TokenStatsPlugin(Star):
    """Token 统计插件，用于统计 LLM API 调用中的 token 使用情况"""
    
    def __init__(self, context: Context):
        super().__init__(context)
        
        # 保存上下文引用
        self.context = context
        
        # 统计数据存储
        self.stats_history: List[Dict[str, Any]] = []
        self.current_stats: Optional[Dict[str, Any]] = None
        
        # 时间跟踪
        self.request_start_time: float = 0.0
        # 当前会话ID
        self.current_session_id: str = ""
        
        # 模型上下文长度配置（模型名 -> 最大上下文长度）
        self.model_max_context: Dict[str, int] = {
            "gpt-4o": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "deepseek-chat": 32768,
            "deepseek-coder": 32768,
            "qwen-turbo": 8192,
            "qwen-plus": 131072,
            "qwen-max": 32768,
            "glm-4": 128000,
            "spark-desk": 8192,
            "ernie-bot": 8192,
            "llama-3": 8192,
            "llama-2": 4096,
            "mistral": 32768,
        }
        
        # 默认上下文长度（当模型未知时使用）
        self.default_max_context: int = 8192
        
        # 统计配置
        self.show_stats_on_response: bool = True  # 是否在每次响应后显示统计
        self.show_detailed_stats: bool = True  # 是否显示详细统计
        self.max_history_size: int = 100  # 最大历史记录数
        
        # 数据文件路径（使用 StarTools 获取规范的数据目录）
        data_dir = StarTools.get_data_dir()
        self.data_file: str = str(data_dir / "token_stats_data.json")
        
        # 加载历史数据
        self._load_history_data()
        
        logger.info("Token 统计插件已初始化")
    
    def _load_history_data(self):
        """加载历史统计数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stats_history = data.get('history', [])
                    # 限制历史记录大小
                    if len(self.stats_history) > self.max_history_size:
                        self.stats_history = self.stats_history[-self.max_history_size:]
                logger.info(f"已加载 {len(self.stats_history)} 条历史统计记录")
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            self.stats_history = []
    
    def _save_history_data(self):
        """保存历史统计数据"""
        try:
            data = {
                'history': self.stats_history,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存历史数据失败: {e}")
    
    def _get_model_max_context(self, model_name: str, provider_id: str = "") -> int:
        """获取模型的最大上下文长度
        
        Args:
            model_name: 模型名称
            provider_id: 提供商ID（从 agent_stats 中获取的实际提供商）
        """
        if not model_name:
            return self.default_max_context
        
        # 1. 首先尝试通过 provider_id 获取提供商配置
        if provider_id:
            try:
                provider = self.context.get_provider_by_id(provider_id)
                if provider and hasattr(provider, 'provider_config'):
                    provider_config = provider.provider_config
                    if isinstance(provider_config, dict):
                        max_context_tokens = provider_config.get('max_context_tokens', 0)
                        if not max_context_tokens or max_context_tokens <= 0:
                            max_context_length = provider_config.get('max_context_length', 0)
                            if max_context_length and max_context_length > 0:
                                logger.info(f"[上下文] 提供商={provider_id}, max_context_length={max_context_length}")
                                return max_context_length
                        
                        if max_context_tokens and max_context_tokens > 0:
                            logger.info(f"[上下文] 提供商={provider_id}, max_context_tokens={max_context_tokens}")
                            return max_context_tokens
            except Exception as e:
                logger.debug(f"[上下文] 通过 provider_id 获取配置失败: {e}")
        
        # 2. 尝试精确匹配硬编码的模型配置
        if model_name in self.model_max_context:
            return self.model_max_context[model_name]
        
        # 3. 尝试模糊匹配硬编码的模型配置
        model_lower = model_name.lower()
        for key, value in self.model_max_context.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return value
        
        # 4. 返回默认值
        return self.default_max_context
    
    def _calculate_stats(self, llm_response: LLMResponse, agent_stats: dict = None) -> Dict[str, Any]:
        """计算统计信息
        
        Args:
            llm_response: LLM 响应对象
            agent_stats: agent_stats 字典（包含累计 token 使用量）
        """
        # 计算响应时间
        end_time = time.time()
        duration = end_time - self.request_start_time if self.request_start_time > 0 else 0
        
        # 初始化变量
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0
        tool_call_tokens = 0
        model_name = "unknown"
        
        # 从 agent_stats 读取累计值（与 WebChat 一致）
        token_usage = agent_stats.get("token_usage", {}) if agent_stats else {}
        
        if token_usage and isinstance(token_usage, dict):
            # 从 agent_stats.token_usage 读取累计值
            prompt_tokens = token_usage.get("input_other", 0) + token_usage.get("input_cached", 0)
            completion_tokens = token_usage.get("output", 0)
            total_tokens = prompt_tokens + completion_tokens
            cached_tokens = token_usage.get("input_cached", 0)
        elif hasattr(llm_response, 'usage') and llm_response.usage:
            # 降级：从 llm_response.usage 读取
            usage = llm_response.usage
            if isinstance(usage, TokenUsage):
                prompt_tokens = usage.input
                completion_tokens = usage.output
                total_tokens = usage.total
                cached_tokens = usage.input_cached
        
        # 从 raw_completion 读取详细信息（reasoning_tokens 等）
        if hasattr(llm_response, 'raw_completion') and llm_response.raw_completion:
            raw = llm_response.raw_completion
            
            # OpenAI 兼容格式
            if hasattr(raw, 'usage') and raw.usage:
                raw_usage = raw.usage
                ctd = getattr(raw_usage, 'completion_tokens_details', None)
                if ctd:
                    reasoning_tokens = getattr(ctd, 'reasoning_tokens', 0) or 0
                    tool_call_tokens = getattr(ctd, 'tool_call_tokens', 0) or 0
            
            # Anthropic 格式
            elif hasattr(raw, 'usage') and hasattr(raw.usage, 'input_tokens'):
                pass  # Anthropic 的 output_tokens 已包含所有输出
            
            # Gemini 格式
            elif hasattr(raw, 'usage_metadata'):
                pass  # Gemini 的 candidates_token_count 已包含所有输出
            
            # Anthropic 格式
            elif hasattr(raw, 'usage') and hasattr(raw.usage, 'input_tokens'):
                raw_usage = raw.usage
                prompt_tokens = getattr(raw_usage, 'input_tokens', 0) or 0
                completion_tokens = getattr(raw_usage, 'output_tokens', 0) or 0
                cached_tokens = getattr(raw_usage, 'cache_read_input_tokens', 0) or 0
                total_tokens = prompt_tokens + completion_tokens
            
            # Gemini 格式
            elif hasattr(raw, 'usage_metadata'):
                raw_usage = raw.usage_metadata
                prompt_tokens = getattr(raw_usage, 'prompt_token_count', 0) or 0
                completion_tokens = getattr(raw_usage, 'candidates_token_count', 0) or 0
                cached_tokens = getattr(raw_usage, 'cached_content_token_count', 0) or 0
                total_tokens = prompt_tokens + completion_tokens
            
            # 字典格式 (通用兼容)
            elif isinstance(raw, dict) and 'usage' in raw:
                raw_usage = raw['usage']
                if isinstance(raw_usage, dict):
                    prompt_tokens = raw_usage.get('prompt_tokens', 0) or 0
                    completion_tokens = raw_usage.get('completion_tokens', 0) or 0
                    total_tokens = raw_usage.get('total_tokens', 0) or 0
                    
                    ptd = raw_usage.get('prompt_tokens_details', {})
                    if isinstance(ptd, dict):
                        cached_tokens = ptd.get('cached_tokens', 0) or 0
                    
                    ctd = raw_usage.get('completion_tokens_details', {})
                    if isinstance(ctd, dict):
                        reasoning_tokens = ctd.get('reasoning_tokens', 0) or 0
                        tool_call_tokens = ctd.get('tool_call_tokens', 0) or 0
        
        # 将 reasoning_tokens 和 tool_call_tokens 加入 completion_tokens
        # 只有当 reasoning_tokens 超过阈值（100）时才加入，避免 API 内部开销被误算
        if reasoning_tokens > 100:
            completion_tokens = completion_tokens + reasoning_tokens
            total_tokens = total_tokens + reasoning_tokens
        if tool_call_tokens > 0:
            completion_tokens = completion_tokens + tool_call_tokens
            total_tokens = total_tokens + tool_call_tokens
        
        # 计算 input_other（新增处理的输入 token）
        input_other = prompt_tokens - cached_tokens if cached_tokens > 0 else prompt_tokens
        
        # 尝试获取模型名称
        # 从 raw_completion 中获取模型信息
        if hasattr(llm_response, 'raw_completion') and llm_response.raw_completion:
            raw = llm_response.raw_completion
            if hasattr(raw, 'model'):
                model_name = raw.model
            elif isinstance(raw, dict) and 'model' in raw:
                model_name = raw['model']
        
        # 如果 total_tokens 为 0，尝试计算
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens
        
        # 计算整体平均速度（含 prefill）
        tokens_per_second = 0.0
        if duration > 0 and completion_tokens > 0:
            tokens_per_second = completion_tokens / duration
        
        # 计算上下文使用率
        max_context = self._get_model_max_context(model_name)
        context_usage = 0.0
        if max_context > 0 and total_tokens > 0:
            context_usage = (total_tokens / max_context) * 100
        
        # 计算 Prefill 速度
        prefill_speed = 0.0
        
        # 构建统计信息（TTFT 将在后续异步填充）
        stats = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'prompt_tokens': prompt_tokens,
            'cached_tokens': cached_tokens,
            'input_other': input_other,
            'completion_tokens': completion_tokens,
            'reasoning_tokens': reasoning_tokens,
            'tool_call_tokens': tool_call_tokens,
            'total_tokens': total_tokens,
            'duration': round(duration, 2),
            'tokens_per_second': round(tokens_per_second, 2),
            'prefill_speed': prefill_speed,  # 稍后根据 TTFT 计算
            'ttft': 0.0,  # 稍后从 event 填充精确值
            'decode_speed': 0.0,  # 稍后计算
            'max_context_length': max_context,
            'context_usage_percent': round(context_usage, 2),
            'start_time': self.request_start_time,
            'end_time': end_time
        }
        
        return stats
    
    def _format_stats_message(self, stats: Dict[str, Any], detailed: bool = True, session_id: str = "") -> str:
        """格式化统计信息为可读消息"""
        # 获取当前会话的历史记录（当前请求已保存到历史中，直接使用）
        session_history = self._get_session_history(session_id)
        
        # 记录调试信息
        logger.debug(f"[会话ID] format_stats: session_id={session_id}, history_count={len(session_history)}, all_count={len(self.stats_history)}")
        
        # 计算累计token统计（只统计当前会话）
        total_prompt = sum(s['prompt_tokens'] for s in session_history)
        total_completion = sum(s['completion_tokens'] for s in session_history)
        total_total = sum(s['total_tokens'] for s in session_history)
        
        if detailed:
            ttft = stats.get('prefill_time', 0)
            decode_speed = stats.get('decode_speed', stats['tokens_per_second'])
            prefill_speed = stats.get('prefill_speed', 0)
            cached_tokens = stats.get('cached_tokens', 0)
            input_other = stats.get('input_other', stats['prompt_tokens'] - cached_tokens if cached_tokens > 0 else stats['prompt_tokens'])
            reasoning_tokens = stats.get('reasoning_tokens', 0)
            tool_call_tokens = stats.get('tool_call_tokens', 0)
            
            # 输入分解
            input_breakdown = ""
            if cached_tokens > 0 or input_other > 0:
                input_breakdown = f"   ├─ 缓存命中: {cached_tokens:,}\n   └─ 新增处理: {input_other:,}"
            
            # 输出分解（只在有显著 reasoning/tool_call tokens 时显示）
            output_breakdown = ""
            # 阈值：只有当 reasoning_tokens > 100 或 tool_call_tokens > 0 时才显示
            if reasoning_tokens > 100 or tool_call_tokens > 0:
                parts = []
                if reasoning_tokens > 100:
                    parts.append(f"深度思考: {reasoning_tokens:,}")
                if tool_call_tokens > 0:
                    parts.append(f"工具调用: {tool_call_tokens:,}")
                output_breakdown = f"\n   └─ 其中{'、'.join(parts)}"
            
            # 时间分解
            time_breakdown = ""
            if ttft > 0:
                decode_time = stats.get('decode_time', stats['duration'] - ttft)
                time_breakdown = f"\n   ├─ Prefill (TTFT): {ttft:.1f}s\n   └─ Decode: {decode_time:.1f}s"
            
            # 速度行
            speed_lines = ""
            if decode_speed > 0:
                speed_lines += f"🚀 解码速度: {decode_speed:.1f} tokens/s"
            if prefill_speed > 0:
                speed_lines += f"\n⚡ Prefill速度: {prefill_speed:.1f} tokens/s"
            
            message = f"""📊 Token 统计信息
━━━━━━━━━━━━━━━━━━
🤖 模型: {stats['model']}
📝 输入 tokens: {stats['prompt_tokens']:,}
{input_breakdown}
💬 输出 tokens: {stats['completion_tokens']:,}{output_breakdown}
📊 总 tokens: {stats['total_tokens']:,}
📈 上下文使用率: {stats['context_usage_percent']:.1f}%
📏 最大上下文: {stats['max_context_length']:,} tokens
━━━━━━━━━━━━━━━━━━
⏱️ 响应时间: {stats['duration']:.1f}s{time_breakdown}
{speed_lines}
━━━━━━━━━━━━━━━━━━
📈 累计统计（当前会话）:
📝 总输入: {total_prompt:,} | 💬 总输出: {total_completion:,} | 📊 总计: {total_total:,}
━━━━━━━━━━━━━━━━━━"""
        else:
            decode_speed = stats.get('decode_speed', stats['tokens_per_second'])
            message = f"📊 Token: {stats['total_tokens']:,} | 生成: {decode_speed:.1f} t/s | 上下文: {stats['context_usage_percent']:.1f}% | 累计: {total_total:,}"
        
        return message
    
    def _get_session_history(self, session_id: str) -> list:
        """获取指定会话的历史记录"""
        if not session_id:
            return []
        # 从stats_history中筛选当前会话的记录（严格匹配）
        return [s for s in self.stats_history if s.get('session_id') == session_id]
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求开始时触发，记录开始时间"""
        self.request_start_time = time.time()
        self.current_stats = None
        # 记录当前会话ID和UMO（用于后续精确查询数据库）
        self.current_session_id = event.get_session_id() if hasattr(event, 'get_session_id') else ""
        self.current_umo = getattr(event, 'unified_msg_origin', '') or ""
        logger.info(f"[会话ID] LLM请求开始: session_id={self.current_session_id}, umo={self.current_umo}")
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM 响应完成时触发，计算统计信息"""
        try:
            # 从 event 获取 agent_stats（包含累计 token 使用量、TTFT、provider 信息）
            agent_stats = event.get_extra("agent_stats") or {}
            
            # 计算统计信息（传入 agent_stats 以获取累计 token 数据）
            stats = self._calculate_stats(resp, agent_stats)
            
            # 获取 TTFT
            ttft = float(agent_stats.get("time_to_first_token", 0) or 0)
            stats['ttft'] = ttft
            
            # 从 agent_stats 获取实际使用的提供商 ID
            provider_id = agent_stats.get("provider_id", "")
            stats['provider_id'] = provider_id
            
            # 重新计算最大上下文长度（使用正确的提供商）
            model_name = stats.get('model', '')
            stats['max_context_length'] = self._get_model_max_context(model_name, provider_id)
            
            # 重新计算上下文使用率
            max_context = stats['max_context_length']
            if max_context > 0 and stats['total_tokens'] > 0:
                stats['context_usage_percent'] = round((stats['total_tokens'] / max_context) * 100, 2)
            
            # 计算真实 decode 速度（排除 prefill 时间）
            duration = stats['duration']
            decode_time = max(0.1, duration - ttft) if ttft > 0 else duration
            decode_speed = stats['completion_tokens'] / decode_time if decode_time > 0 else 0
            stats['decode_speed'] = round(decode_speed, 2)
            stats['prefill_time'] = round(ttft, 2)
            stats['decode_time'] = round(decode_time, 2)
            
            # 计算 Prefill 速度（新增处理的输入 token / TTFT）
            input_other = stats.get('input_other', 0)
            if ttft > 0 and input_other > 0:
                stats['prefill_speed'] = round(input_other / ttft, 1)
            else:
                stats['prefill_speed'] = 0.0
            
            # 添加会话ID到统计信息
            stats['session_id'] = self.current_session_id
            self.current_stats = stats
            
            # 添加到历史记录
            self.stats_history.append(stats)
            
            # 限制历史记录大小
            if len(self.stats_history) > self.max_history_size:
                self.stats_history = self.stats_history[-self.max_history_size:]
            
            # 保存历史数据
            self._save_history_data()
            
            # 如果需要显示统计信息
            if self.show_stats_on_response:
                # 使用会话ID过滤统计，避免将本请求的统计信息计入累计
                session_id = self.current_session_id
                message = self._format_stats_message(stats, self.show_detailed_stats, session_id)
                
                # 记录日志
                logger.info(f"Token 统计: {stats['model']} - {stats['total_tokens']} tokens, "
                           f"生成: {stats['decode_speed']:.1f} t/s, "
                           f"TTFT: {stats['prefill_time']:.1f}s, "
                           f"上下文使用率 {stats['context_usage_percent']:.1f}%")
                
                # 尝试发送消息到聊天平台
                try:
                    # 使用事件装饰器添加统计信息到消息链
                    result = event.get_result()
                    if result and hasattr(result, 'chain'):
                        # 在现有消息链末尾添加统计信息
                        result.chain.append(Comp.Plain("\n\n" + message))
                except Exception as e:
                    logger.warning(f"发送统计信息失败: {e}")
            
            logger.debug(f"Token 统计完成: {stats}")
            
        except Exception as e:
            logger.error(f"处理 LLM 响应时出错: {e}")
    
    @filter.command("token_stats", alias={'token统计', '统计'})
    async def show_stats_command(self, event: AstrMessageEvent, sub_command: str = ""):
        """显示 token 统计信息的命令"""
        try:
            # 从事件中获取当前会话ID
            session_id = event.get_session_id() if hasattr(event, 'get_session_id') else ""
            logger.info(f"[会话ID] token_stats命令: session_id={session_id}, stats_history长度={len(self.stats_history)}")
            if sub_command == "history" or sub_command == "历史":
                # 获取当前会话的历史记录
                session_history = self._get_session_history(session_id)
                
                if not session_history:
                    yield event.plain_result("📊 暂无当前会话历史统计记录")
                    return
                
                # 计算平均统计（按会话ID过滤）
                total_stats = len(session_history)
                avg_prompt = sum(s['prompt_tokens'] for s in session_history) / total_stats
                avg_completion = sum(s['completion_tokens'] for s in session_history) / total_stats
                avg_total = sum(s['total_tokens'] for s in session_history) / total_stats
                avg_speed = sum(s['tokens_per_second'] for s in session_history) / total_stats
                avg_context = sum(s['context_usage_percent'] for s in session_history) / total_stats
                
                message = f"""📊 Token 统计历史摘要（当前会话）
━━━━━━━━━━━━━━━━━━
📈 统计次数: {total_stats}
📝 平均 Prompt tokens: {avg_prompt:,.0f}
💬 平均 Completion tokens: {avg_completion:,.0f}
📊 平均 Total tokens: {avg_total:,.0f}
⚡ 平均生成速度: {avg_speed:.1f} tokens/s
📈 平均上下文使用率: {avg_context:.1f}%
━━━━━━━━━━━━━━━━━━"""
                
                yield event.plain_result(message)
                
            elif sub_command == "avg" or sub_command == "平均":
                # 显示平均统计（最近10次）
                recent_stats = self.stats_history[-10:] if len(self.stats_history) >= 10 else self.stats_history
                
                if not recent_stats:
                    yield event.plain_result("📊 暂无统计记录")
                    return
                
                total_stats = len(recent_stats)
                avg_prompt = sum(s['prompt_tokens'] for s in recent_stats) / total_stats
                avg_completion = sum(s['completion_tokens'] for s in recent_stats) / total_stats
                avg_total = sum(s['total_tokens'] for s in recent_stats) / total_stats
                avg_speed = sum(s['tokens_per_second'] for s in recent_stats) / total_stats
                avg_context = sum(s['context_usage_percent'] for s in recent_stats) / total_stats
                
                message = f"""📊 最近 {total_stats} 次统计平均值
━━━━━━━━━━━━━━━━━━
📝 平均 Prompt tokens: {avg_prompt:,.0f}
💬 平均 Completion tokens: {avg_completion:,.0f}
📊 平均 Total tokens: {avg_total:,.0f}
⚡ 平均生成速度: {avg_speed:.1f} tokens/s
📈 平均上下文使用率: {avg_context:.1f}%
━━━━━━━━━━━━━━━━━━"""
                
                yield event.plain_result(message)
                
            elif sub_command == "clear" or sub_command == "清空":
                # 清空历史记录
                self.stats_history.clear()
                self._save_history_data()
                yield event.plain_result("✅ 历史统计记录已清空")
                
            elif sub_command == "config" or sub_command == "配置":
                # 显示当前配置
                message = f"""⚙️ Token 统计插件配置
━━━━━━━━━━━━━━━━━━
📊 响应后显示统计: {'是' if self.show_stats_on_response else '否'}
📋 显示详细统计: {'是' if self.show_detailed_stats else '否'}
📁 最大历史记录: {self.max_history_size}
🤖 默认上下文长度: {self.default_max_context:,}
━━━━━━━━━━━━━━━━━━"""
                
                yield event.plain_result(message)
                
            else:
                # 显示最近一次统计（按会话ID）
                if self.current_stats:
                    message = self._format_stats_message(self.current_stats, self.show_detailed_stats, session_id)
                    yield event.plain_result(message)
                else:
                    # 查找当前会话的最后一次统计
                    session_history = self._get_session_history(session_id)
                    if session_history:
                        last_stats = session_history[-1]
                        message = self._format_stats_message(last_stats, self.show_detailed_stats, session_id)
                        yield event.plain_result(message)
                    else:
                        yield event.plain_result("📊 暂无统计记录，请先进行一次对话")
                    
        except Exception as e:
            logger.error(f"执行 token_stats 命令时出错: {e}")
            yield event.plain_result(f"❌ 执行命令时出错: {e}")
    
    @filter.command("token_config", alias={'token配置'})
    async def config_command(self, event: AstrMessageEvent, setting: str = "", value: str = ""):
        """配置 token 统计插件"""
        try:
            if not setting:
                # 显示当前配置
                message = f"""⚙️ Token 统计插件配置
━━━━━━━━━━━━━━━━━━
📊 响应后显示统计: {'是' if self.show_stats_on_response else '否'}
📋 显示详细统计: {'是' if self.show_detailed_stats else '否'}
📁 最大历史记录: {self.max_history_size}
🤖 默认上下文长度: {self.default_max_context:,}

使用方法:
/token_config show_stats on/off  # 开启/关闭响应后显示统计
/token_config detailed on/off    # 开启/关闭详细统计
/token_config max_history 100    # 设置最大历史记录数
/token_config default_context 8192  # 设置默认上下文长度
━━━━━━━━━━━━━━━━━━"""
                yield event.plain_result(message)
                return
            
            # 处理配置更新
            if setting == "show_stats":
                if value.lower() in ["on", "true", "1", "是"]:
                    self.show_stats_on_response = True
                    yield event.plain_result("✅ 已开启响应后显示统计")
                elif value.lower() in ["off", "false", "0", "否"]:
                    self.show_stats_on_response = False
                    yield event.plain_result("✅ 已关闭响应后显示统计")
                else:
                    yield event.plain_result("❌ 无效的值，请使用 on/off")
                    
            elif setting == "detailed":
                if value.lower() in ["on", "true", "1", "是"]:
                    self.show_detailed_stats = True
                    yield event.plain_result("✅ 已开启详细统计显示")
                elif value.lower() in ["off", "false", "0", "否"]:
                    self.show_detailed_stats = False
                    yield event.plain_result("✅ 已关闭详细统计显示")
                else:
                    yield event.plain_result("❌ 无效的值，请使用 on/off")
                    
            elif setting == "max_history":
                try:
                    max_val = int(value)
                    if max_val > 0:
                        self.max_history_size = max_val
                        yield event.plain_result(f"✅ 已设置最大历史记录数为 {max_val}")
                    else:
                        yield event.plain_result("❌ 值必须大于 0")
                except ValueError:
                    yield event.plain_result("❌ 无效的数字")
                    
            elif setting == "default_context":
                try:
                    context_val = int(value)
                    if context_val > 0:
                        self.default_max_context = context_val
                        yield event.plain_result(f"✅ 已设置默认上下文长度为 {context_val}")
                    else:
                        yield event.plain_result("❌ 值必须大于 0")
                except ValueError:
                    yield event.plain_result("❌ 无效的数字")
                    
            else:
                yield event.plain_result(f"❌ 未知的配置项: {setting}")
                
        except Exception as e:
            logger.error(f"执行配置命令时出错: {e}")
            yield event.plain_result(f"❌ 执行命令时出错: {e}")
    
    async def terminate(self):
        """插件卸载时调用"""
        # 保存最终数据
        self._save_history_data()
        logger.info("Token 统计插件已卸载")