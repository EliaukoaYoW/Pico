"""
Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、memory、相关笔记、历史
以及当前用户请求送进模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

DEFAULT_TOTAL_BUDGET = 12000  # 整个 Prompt 允许的最大字符数
DEFAULT_SECTION_BUDGETS = {
    "prefix": 3600,          # 系统指令 + 工作区快照
    "memory": 1600,          # 工作记忆（任务摘要 + 最近读过的文件）
    "relevant_memory": 1200, # 根据当前请求召回的相关历史笔记
    "history": 5200          # 本次会话的历史记录
}

# 每个部分的最小预算，防止模型输出为空
DEFAULT_SECTION_FLOORS = {
    "prefix": 1200,
    "memory": 400,
    "relevant_memory": 300,
    "history": 1500
}
# 当 Prompt 超预算时的压缩顺序
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")
# 拼接 Prompt 时各 Section 的排列顺序（从上到下）
SECTION_ORDER = ("prefix", "memory", "relevant_memory", "history", "current_request")
CURRENT_REQUEST_SECTION = "current_request"  # 当前用户的请求环节
RELEVANT_MEMORY_LIMIT = 3                    # 最多召回 3 条相关历史笔记

def _tail_clip(text: Any, limit: int) -> str:
    """ 
    对文本进行尾部裁剪，确保不超过指定长度
    输入: 
    - text: 待裁剪的文本
    - limit: 最大允许的字符数
    输出: 裁剪后的文本
    """
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[:limit-3] + "..."

@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self) -> int:
        return len(self.raw)
    
    @property
    def rendered_chars(self) -> int:
        return len(self.rendered)

class ContextManager:
    """ 
    上下文管理器 负责根据预算组装 Prompt 
    组装顺序: prefix -> memory -> relevant_memory -> history -> current_request
    """
    def __init__(
        self,
        agent,
        total_budget = DEFAULT_TOTAL_BUDGET,    # 整个 Prompt 允许的最大字符数
        section_budgets = None,                 # 每个部分的预算
        section_floors = None,                  # 每个部分的最小预算
        reduction_order = None                  # 当 Prompt 超预算时的压缩顺序
    ):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()
        self.reduction_order = tuple(reduction_order or DEFAULT_REDUCTION_ORDER)
        
    def build(self, user_message: Any) -> tuple[str, dict[str, Any]]:
        """ 功能: 按预算组装一轮完整 prompt。
        用户请求 -> 收集上下文（各个 section 的内容 section_texts）-> 裁剪上下文（根据预算）-> 组装 prompt

        为什么存在：仅靠用户这一轮输入，模型并不知道当前仓库状态、会话里已经读过什么、哪些旧信息还值得继续参考。
        这个函数负责把“稳定基线 + 工作记忆 + 相关笔记 + 历史 + 当前请求”拼成真正发给模型的 prompt。

        输入 / 输出：
        - 输入：`user_message`，也就是用户当前这一轮的新请求。
        - 输出：`(prompt, metadata)`。
          `prompt` 是最终发送给模型的文本；
          `metadata` 记录了每个 section 的原始长度、裁剪后的长度、是否触发了预算收缩等信息，
          后续会进入 trace/report，便于解释这轮 prompt是怎么被拼出来的。

        在 agent 链路里的位置：
        它位于 `Pico.ask()` 的每轮模型调用之前，是“真正发请求给模型”的最后一道组装工序。
        `WorkspaceContext` 提供稳定前缀，
        `LayeredMemory`提供工作记忆，这个函数则把它们和当前请求合成一份可控大小的 prompt。
        """
        user_message = str(user_message)
        self.section_floors = self._compute_section_floors()
        memory_enabled = True
        relevant_memory_enabled = True
        context_reduction_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "memory": "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text()),
            "history": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}"
        }
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["prefix"] = section_texts["prefix"] + "\n\n" + checkpoint_text
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit = RELEVANT_MEMORY_LIMIT)

        # 用于功能测试
        if not context_reduction_enabled:
            rendered = self._render_sections_without_reduction(section_texts, selected_notes = selected_notes)
            prompt = self._assemble_prompt(rendered)
            metadata = self._metadata(
                prompt=prompt,
                rendered=rendered,
                budgets={section: render.budget for section, render in rendered.items() if section != CURRENT_REQUEST_SECTION},
                reduction_log=[],
                selected_notes=selected_notes,
                user_message=user_message,
                section_texts=section_texts,
            )
            return prompt, metadata

        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes = selected_notes)
        prompt = self._assemble_prompt(rendered)
        reduction_log = []

        # 超预算时，根据压缩顺序压缩每个 section 直至不超过 total_budget
        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                reduction_log.append(
                    {
                        "section": section,
                        "before_chars": current_budget,
                        "after_chars": new_budget,
                        "overflow_chars": overflow
                    }
                )
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes = selected_notes)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break
        metadata = self._metadata(
            prompt = prompt,
            rendered = rendered,
            budgets = budgets,
            reduction_log = reduction_log,
            selected_notes = selected_notes,
            user_message = user_message,
            section_texts = section_texts
        )
        return prompt, metadata
        
    def _render_sections_without_reduction(self, section_texts: dict[str, str], selected_notes: list[dict[str, Any]] | None = None) -> dict[str, SectionRender]:
        """
        不做任何压缩，直接暂渲染各 section。
        输入: section_texts（各 section 的原始文本）、selected_notes（已召回的相关笔记）
        输出: 各 section 对应的 SectionRender
        用途: 当 context_reduction 功能被禁用时（测试 / 实验用）
        """
        selected_notes = selected_notes or []
        relevant_lines = ["Relevant Memory:"]
        if selected_notes:
            relevant_lines.extend(f"- {note['text']}" for note in selected_notes)
        else:
            relevant_lines.append("- none")
        relevant_raw = "\n".join(relevant_lines)
        history = list(getattr(self.agent, "session", {}).get("history", []))
        history_raw = self._render_history_text(history)
        return {
            "prefix": SectionRender(raw = section_texts["prefix"], budget = len(section_texts["prefix"]), rendered = section_texts["prefix"], details = {}),
            "memory": SectionRender(raw = section_texts["memory"], budget = len(section_texts["memory"]), rendered = section_texts["memory"], details = {}),
            "relevant_memory": SectionRender(
                raw = relevant_raw,
                budget = len(relevant_raw),
                rendered = relevant_raw,
                details = {
                    "selected_notes": [note["text"] for note in selected_notes],
                    "rendered_notes": [note["text"] for note in selected_notes],
                    "selected_count": len(selected_notes),
                    "rendered_count": len(selected_notes),
                    "note_budget": 0,
                },
            ),
            "history": SectionRender(raw = history_raw, budget = len(history_raw), rendered = history_raw, details = {"rendered_entries": []}),
            CURRENT_REQUEST_SECTION: SectionRender(
                raw=section_texts[CURRENT_REQUEST_SECTION],
                budget=0,
                rendered=section_texts[CURRENT_REQUEST_SECTION],
                details={},
            ),
        }


    def _compute_section_floors(self) -> dict[str, int]:
        """
        计算每个 section 压缩时的最低保障线（floor）
        输入: 无
        输出: 各 section 对应的最低保障线（floor）
        用途: 当 Prompt 超预算时，根据压缩顺序压缩每个 section 直至不超过 total_budget
        """
        floors = {
            section: max(20, int(budget) // 4)
            for section, budget in self.section_budgets.items()
        }
        floors.update(self._section_floor_overrides)
        return floors
    
    def _render_sections(self, section_texts: dict[str, str], budgets: dict[str, int], selected_notes: list[dict[str, Any]] | None = None) -> dict[str, SectionRender]:
        """
        渲染各 section，根据预算压缩。
        输入: section_texts（各 section 的原始文本）、budgets（各 section 的预算）、selected_notes（已召回的相关笔记）
        输出: 各 section 对应的 SectionRender
        用途: 当 Prompt 超预算时，根据压缩顺序压缩每个 section 直至不超过 total_budget
        """
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                # 当前请求永不裁剪 直接渲染
                raw = section_texts[section]
                rendered[section] = SectionRender(raw = raw, budget = 0, rendered = raw, details = {})
            elif section == "relevant_memory":
                # 相关记忆 -> 有独立的渲染逻辑
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))
            elif section == "history":
                # 历史记录 -> 有独立的渲染逻辑
                rendered[section] = self._render_history_section(int(budget or 0))
            else:
                # prefix 和 memory 直接裁剪
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget)) if budget else raw
                rendered[section] = SectionRender(raw = raw, budget = int(budget or 0),  rendered = rendered_text, details = {})
        return rendered
    
    def _render_relevant_memory(self, selected_notes: list[dict[str, Any]], budget: int) -> SectionRender:
        """
        渲染相关记忆 section 并根据预算压缩。
        输入: selected_notes（已召回的相关笔记）、budget（相关记忆 section 的预算）
        输出: 相关记忆 section 的 SectionRender
        用途: 当 Prompt 超预算时，根据压缩顺序压缩相关记忆 section 直至不超过 total_budget
        """
        header = "Relevant Memory:"
        note_texts = [str(note.get("text", "")) for note in selected_notes if str(note.get("text", "")).strip()]
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])
        if not note_texts:
            rendered = raw
            return SectionRender(
                raw=raw, budget=budget, rendered=rendered, 
                details={
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0
                }
            )
        
        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            # 让每条 note 平分这一段的预算，避免一条超长笔记把其他笔记都挤掉
            # 如果整体还超出预算，就减少每个笔记的预算，直到符合预算
            rendered_notes = [_tail_clip(text, per_note_budget) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1
        
        if len(rendered) > budget and budget > 0:
            # 如果整体还超出预算，就对整个段做最后的整体裁剪
            rendered = _tail_clip(raw, budget)
            rendered_notes = [rendered]
            
        return SectionRender(
            raw = raw, budget = budget, rendered = rendered, 
            details = {
                "selected_notes": note_texts,
                "rendered_notes": rendered_notes,
                "selected_count": len(note_texts),
                "rendered_count": len(rendered_notes),
                "note_budget": per_note_budget
            }
        )

    def _per_note_budget(self, budget: int, note_count: int, header: str) -> int:
        """
        计算每条 note 的平分到的字符预算
        输入: budget（总预算）、note_count（笔记条数）、header（标题）
        输出: 每条 note 的最大字符预算（最小为 1 个字符）
        公式: max(1, (budget - len(header) - 3 * note_count) // note_count)
        """
        if note_count <= 0:
            return 0
        overhead = len(header) + 3 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)
    
    def _render_history_section(self, budget: int) -> SectionRender:
        """
        将历史记录 section 渲染进 Prompt，优先保留最近 6 条，旧条目被压缩或折叠。
        输入: budget（分配给 历史记录 section 的字符预算）
        输出: 历史记录 section 的 SectionRender
        """
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self._raw_history_text(history)
        if not history:
            rendered = "Transcript:\n- empty"
            return SectionRender(
                raw = raw, budget = budget, rendered = rendered, 
                details = {
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0
                }
            )
        # 优先保留最近的历史记录
        recent_window = 6
        recent_start = max(0, len(history) - recent_window)
        history_entries, history_details = self._compressed_history_entries(history, recent_start)
        rendered_entries = []
        for entry in reversed(history_entries):
            recent = bool(entry.get("recent", False))
            candidate_lines = list(entry.get("lines", []))
            candidate_entries = candidate_lines + rendered_entries
            candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
            if len(candidate_rendered) <= budget:
                rendered_entries = candidate_entries
                continue
            if recent:
                available = budget - len("Transcript:")
                if rendered_entries:
                    available -= sum(len(line) + 1 for line in rendered_entries)
                available = max(20, available - 1)
                candidate_lines = [_tail_clip(line, available) for line in candidate_lines]
                candidate_entries = candidate_lines + rendered_entries
                candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
                if len(candidate_rendered) <= budget:
                    rendered_entries = candidate_entries
            else:
                smaller_lines = [_tail_clip(line, 20) for line in candidate_lines]
                smaller_entries = smaller_lines + rendered_entries
                smaller_rendered = "\n".join(["Transcript:", *smaller_entries])
                if len(smaller_rendered) <= budget:
                    rendered_entries = smaller_entries
        rendered = "\n".join(["Transcript:", *rendered_entries])

        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)

        return SectionRender(
            raw = raw, budget = budget, rendered = rendered,
            details = {
                "recent_window": recent_window,
                "recent_start": recent_start,
                "rendered_entries": rendered_entries,
                **history_details,
            },
        )

    def _compressed_history_entries(self, history: list[dict[str, Any]], recent_start: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """
        将历史条目分成「最近」和「较旧」两类，分别处理:
        - 最近条目：保留完整内容（每条最多 900 字符）
        - 较旧的 read_file：用记忆中的文件摘要替代，重复的直接丢弃
        - 较旧的其他工具：压缩成一行摘要
        - 较旧的用户/助手消息：压缩到 60 字符
        输入: history（历史记录列表）、recent_start（最近记录起始索引）
        输出: 压缩后的历史记录列表 entries、压缩详情 details
        """
        entries = []
        seen_older_reads = set()
        details = {
            "older_entries_count": 0,
            "collapsed_duplicate_reads": 0,
            "reused_file_summary_count": 0,
            "summarized_tool_count": 0
        }
        
        for index, item in enumerate(history):
            recent = index >= recent_start
            if recent:
                line_limit = 900
                entries.append(
                    {
                        "recent": True,
                        "lines": self._render_history_item(item, line_limit)
                    }
                )
                continue
            if item["role"] == "tool" and item["name"] == "read_file":
                path = str(item["args"].get("path", "")).strip()
                if path in seen_older_reads:
                    details["collapsed_duplicate_reads"] += 1
                    continue
                seen_older_reads.add(path)
                summary = self._reusable_file_summary(path)
                if summary:
                    entries.append({"recent":False, "lines": [f"{path} -> {summary}"]})
                    details["older_entries_count"] += 1
                    details["reused_file_summary_count"] += 1
                    continue
            
            if item["role"] == "tool":
                summary_line = self._summarize_old_tool_item(item)
                entries.append({"recent": False, "lines": [summary_line]})
                details["older_entries_count"] += 1
                details["summarized_tool_count"] += 1
                continue
            
            entries.append({"recent": False, "lines": self._render_history_item(item, 60)})
        return entries, details
    

    def _reusable_file_summary(self, path: str) -> str:
        """
        从记忆中获取文件的已有摘要，用于代替历史记录中的 read_file 操作，如果不存在则返回空字符串。
        输入: path（文件路径）
        输出: 文件摘要（如果存在）
        """
        memory = getattr(self.agent, "memory", None)
        if memory is None or not hasattr(memory, "to_dict"):
            return ""
        snapshot = memory.to_dict()
        summary = snapshot.get("file_summaries", {}).get(str(path), "")
        if not summary:
            return ""
        return str(summary.get("summary", "")).strip()
    
    def _summarize_old_tool_item(self, item: dict[str, Any]) -> str:
        """
        将一条较旧的工具调用记录压缩成单行摘要。
        run_shell：格式为「命令 -> 前三行输出」
        其他工具：直接截断到 60 字符
        输入：item（一条历史记录字典）
        输出：单行摘要字符串
        """
        if item["name"] == "run_shell":
            command = str(item["args"].get("command", "")).strip() or "shell"
            lines = [line.strip() for line in str(item.get("content","")).splitlines() if line.strip()]
            summary = " | ".join(lines[:3]) if lines else "(empty)"
            return f"{command} -> {summary}"
        return self._render_history_item(item, 60)[0]
    
    def _raw_history_text(self, history: list[dict[str, Any]]) -> str:
        """
        将完整历史列表转成未压缩的纯文本（用于统计原始长度）。
        输入：history（历史列表）
        输出：格式化后的 Transcript 字符串
        """
        if not history:
            return "Transcript:\n- empty"
        lines = []
        for item in history:
            if item["role"] == "tool":
                lines.append(f"[assistant] <tool>{{\"name\":\"{item['name']}\",\"args\":{json.dumps(item['args'], sort_keys=True)}}}</tool>")
                lines.append(f"[system] Tool result:\n{item['content']}")
            else:
                lines.append(f"[{item['role']}] {item['content']}")
        return "\n".join(["Transcript:", *lines])
    

    def _render_history_item(self, item: dict[str, Any], line_limit: int) -> list[str]:
        """
        将单条历史记录渲染成行列表，并按 line_limit 裁剪内容。
        输入：item（单条历史记录字典）、line_limit（每行最大字符数）
        输出：格式化后的列表，包含前缀（如 "[tool: 命令]"）和内容（如 "命令输出"）
        """
        if item["role"] == "tool":
            prefix = f"[assistant] <tool>{{\"name\":\"{item['name']}\",\"args\":{json.dumps(item['args'], sort_keys=True)}}}</tool>"
            content = f"[system] Tool result:\n{_tail_clip(item['content'], max(20, line_limit))}"
            return [prefix, content]
        return [f"[{item['role']}] {_tail_clip(item['content'], line_limit)}"]
    
    def _assemble_prompt(self, rendered: dict[str, SectionRender]) -> str:
        # 组装Prompt 其顺序是刻意设计的：稳定规则放前面，最新请求放最后。
        return "\n\n".join(
            [
                rendered["prefix"].rendered,
                rendered["memory"].rendered,
                rendered["relevant_memory"].rendered,
                rendered["history"].rendered,
                rendered[CURRENT_REQUEST_SECTION].rendered,
            ]
        ).strip()

    
    def _metadata(
        self, 
        prompt: str, 
        rendered: dict[str, SectionRender], 
        budgets: dict[str, int], 
        reduction_log: list[dict[str, Any]], 
        selected_notes: list[dict[str, Any]], 
        user_message: str, 
        section_texts: dict[str, str]
    ) -> dict[str, Any]:
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": int(budgets.get(section, 0)),
                "rendered_chars": rendered[section].rendered_chars
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
        }
        return {
            "prompt_chars": len(prompt),
            "prompt_budget_chars": self.total_budget,
            "prompt_over_budget": len(prompt) > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
                for section in SECTION_ORDER
            },
            "sections": section_metadata,
            "budget_reductions": reduction_log,
            "reduction_order": list(self.reduction_order),
            "relevant_memory": {
                "limit": RELEVANT_MEMORY_LIMIT,
                "selected_count": len(selected_notes),
                "selected_notes": [note["text"] for note in selected_notes],
                "selected_source": [str(note.get("source", "")) for note in selected_notes],
                "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],
                "selected_durable_count": sum(
                    1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
                ),
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),
            },
            "history": {
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),
                "reused_file_summary_count": int(rendered["history"].details.get("reused_file_summary_count", 0)),
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),
            },
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            }
        }
