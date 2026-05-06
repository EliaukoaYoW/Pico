"""
命令行入口
"""

import argparse
from json import load
import os
import shutil
import sys
import textwrap

from dotenv import load_dotenv
from models import OpenAICompatibleModelClient, SiliconflowModelClient
from runtime import Pico, SessionStore
from workspace import WorkspaceContext, middle

load_dotenv()

DEFAULT_SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "RIGHT_CODES_API_KEY",
    "SILICONFLOW_API_KEY"
    "GITHUB_PAT",
    "GH_PAT"
)

WELCOME_ART = (
    "      Hi!",
    "       /",
    "   (\"\\(•-•)/\")",
    "     \\     /",
    "      o   o",
    "     (     )",
    "      \\_T_/",
)
WELCOME_NAME = "pico"
WELCOME_SUBTITLE = "local coding agent"
WELCOME_STATUS = "calm shell, ready for work"

HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()

DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"

DEFAULT_SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


LEGACY_SECRET_ENV_NAMES_VAR = "MINI_CODING_AGENT_SECRET_ENV_NAMES"
SECRET_ENV_NAMES_VAR = "PICO_SECRET_ENV_NAMES"



def _effective_model(args, provider="openai"):
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        return explicit_model
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL")
        if model:
            return model
        return DEFAULT_OPENAI_MODEL
    if provider == "siliconflow":
        model = os.environ.get("SILICONFLOW_MODEL")
        if model:
            return model
        return DEFAULT_SILICONFLOW_MODEL

def _first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""

def _configured_secret_names(args):
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")
    if not extra_names.strip():
        extra_names = os.environ.get(LEGACY_SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    return sorted(configured_secret_names)

def _build_model_client(args):
    provider = getattr(args, "provider", "openai")
    if provider == "openai":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or os.environ.get("OPENAI_API_BASE") or DEFAULT_OPENAI_BASE_URL
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "siliconflow":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or os.environ.get("SILICONFLOW_API_BASE") or DEFAULT_SILICONFLOW_BASE_URL
        api_key = _first_env("SILICONFLOW_API_KEY", "")
        return SiliconflowModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "siliconflow-timeout", getattr(args, "ollama_timeout", 300)),
        )
    # 待补充 Anthropic Provider 和 Ollama



def build_welcome(agent, model, host):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width-4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center(WELCOME_NAME),
            center(WELCOME_SUBTITLE),
            center(WELCOME_STATUS),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])

def build_agent(args):
    """
    根据 CLI 参数装配出一个可运行的 Pico 实例。
    为什么存在：
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：
    model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把“启动参数”翻译成“agent 运行现场”。

    输入 / 输出：
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `Pico`，或一个从旧 session 恢复出来的 `Pico`

    在 agent 链路里的位置：
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，
    得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会落到 `ask()`。
    """
    # 这里是 CLI 到 runtime 的装配点: 先整理 secret 名单，再采集工作区快照，
    # 随后决定是恢复旧 session 还是创建一个新的 Pico 实例
    configured_secret_names = _configured_secret_names(args)
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(workspace.repo_root + "/.pico/sessions")
    model = _build_model_client(args)
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return Pico.from_session(
            model_client = model,
            workspace = workspace,
            session_store = store,
            session_id = session_id,
            approval_policy = args.approval,
            max_steps = args.max_steps,
            max_new_tokens = args.max_new_tokens,
            secret_env_names = configured_secret_names
        )
    return Pico(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names
    )

def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for Ollama, OpenAI-compatible, or Anthropic-compatible models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--provider", choices=("ollama", "openai", "anthropic","siliconflow"), default="siliconflow", help="Model backend to use.")
    parser.add_argument("--model", default=None, help="Model name override. Defaults to qwen3.5:4b for Ollama, OPENAI_MODEL for openai, and ANTHROPIC_MODEL for anthropic when set.",)
    parser.add_argument("--host", default="DEFAULT_OLLAMA_HOST", help="Ollama server URL.")
    parser.add_argument("--base-url", default=None, help="Provider API base URL for openai or anthropic.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--siliconflow-timeout", type=int, default=300, help="SiliconFlow--compatible request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    return parser

def main(argv = None):
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)
    model = getattr(agent.model_client, "model", getattr(args, "model", DEFAULT_OPENAI_MODEL))
    host = getattr(agent.model_client, "host", getattr(agent.model_client, "base_url", getattr(args, "host", "")))
    print(build_welcome(agent, model, host))

    if args.prompt:
        # 单次会话模式：只跑一次 ask，不进入 REPL 循环
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                print(agent.ask(prompt))
            except RuntimeError as e:
                print(str(e), file = sys.stderr)
                return 1
        return 0

    while True:
        # 交互模式
        try:
            user_input = input("\nPico > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            result = agent.ask(user_input)
            for char in result:
                print(char, end="", flush=True)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)