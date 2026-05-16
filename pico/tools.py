"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import shutil
import subprocess
import textwrap
from functools import partial

from workspace import IGNORED_PATH_NAMES, clip

BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace."
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a UTF-8 file by line range."
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback."
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root."
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file."
    },
    "patch_file":{
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace one exact text block in a file."
    },
    "list_skills": {
        "schema": {},
        "risky": False,
        "description": "List all available skill documents in the .pico/skills/ directory.",
    },
    "read_skill": {
        "schema": {"name": "str"},
        "risky": False,
        "description": "Read a specific skill document by name.",
    },
}

DELEGATE_TOOL_SPECS = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "risky": True,
    "description": "Ask a bounded read-only sub agent to investigate."
}

TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files", "args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
    "list_skills": '<tool>{"name":"list_skills","args":{}}</tool>',
    "read_skill": '<tool>{"name":"read_skill","args":{"name":"copyright_generator"}}</tool>'
}


def build_tool_registry(agent):
    # 工具不是动态发现的，而是显式注册的。这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], agent)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，就连 delegate 这个工具都不再暴露给模型。
    if agent.depth < agent.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPECS, "run": partial(tool_delegate, agent)}
    return tools

def tool_example(name):
    return TOOL_EXAMPLES.get(name,"")

def validate_tool(agent, name, args):
    args = args or {}

    if name == "list_files":
        path = agent.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return
    
    if name == "read_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        agent.path(args.get("path", "."))
        return
    
    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be between 1 and 120 seconds")
        return
    
    if name == "write_file":
        path = agent.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("content must be provided")
        return

    if name == "patch_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        return
    
    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        return
    
    if name == "list_skills":
        return
    
    if name == "read_skill":
        skill_name = str(args.get("name", "")).strip()
        if not skill_name:
            raise ValueError("skill name must not be empty")

        if ".." in skill_name or "/" in skill_name or "\\" in skill_name:
            raise ValueError("invalid skill name")
        return
    

def tool_list_files(agent, args):
    path = agent.path(args.get("path","."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    
    entries = [
        item for item in sorted(path.iterdir(), key = lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(agent.root)}")
    return "\n".join(lines) or "(empty)"

def tool_read_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(agent.root)}\n{body}"

def tool_search(agent, args):
    pattern = str(args.get("pattern","")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path","."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=agent.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"
    
    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"

def tool_run_shell(agent, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be between 1 and 120 seconds")
    result = subprocess.run(
        command,
        cwd = agent.root,
        shell = True,
        capture_output = True,
        text = True,
        timeout = timeout,
        env = agent.shell_env()
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(agent, args):
    path = agent.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(agent.root)} ({len(content)} chars)"

def tool_patch_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(agent.root)} ({len(text)} chars)"

def tool_delegate(agent, args):
    if agent.depth >= agent.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")
    
    from .runtime import Pico

    child = Pico(
        model_client = agent.model_client,
        workspace = agent.workspace,
        session_store = agent.session_store,
        run_store = agent.run_store,
        approval_policy = "never",
        max_steps = int(args.get("max_steps", 3)),
        max_new_tokens = agent.max_new_tokens,
        depth = agent.depth + 1,
        max_depth = agent.max_depth,
        read_only = True,
        secret_env_names = agent.secret_env_names,
        shell_env_allowlist = agent.shell_env_allowlist
    )
    child.session["memory"]["task"] = task
    child.session["memory"]["notes"] = [clip(agent.history_text(), 300)]
    return "delegate_result:\n" + child.ask(task)


def tool_list_skills(agent, args):
    skills_dir = agent.root / ".pico" / "skills"
    if not skills_dir.exists() or not skills_dir.is_dir():
        return "(no skills available)"
    
    skills = []
    for item in skills_dir.iterdir():
        if item.is_file() and item.name.endswith(".md"):
            skills.append(item.stem)
        elif item.is_dir():
            if (item / "SKILL.md").is_file() or (item / "README.md").is_file():
                skills.append(f"{item.name} (package)")
                
    if not skills:
        return "(no skills available)"
    
    return "Available skills:\n" + "\n".join(f"- {s}" for s in sorted(skills))


def tool_read_skill(agent, args):
    raw_name = str(args.get("name", "")).strip()
    skill_name = raw_name.replace(" (package)", "")
    if skill_name.endswith(".md"):
        skill_name = skill_name[:-3]
        
    skills_dir = agent.root / ".pico" / "skills"
    
    dir_path = skills_dir / skill_name
    if dir_path.is_dir():
        md_path = dir_path / "SKILL.md"
        if not md_path.is_file():
            md_path = dir_path / "README.md"
        
        if md_path.is_file():
            content = md_path.read_text(encoding="utf-8", errors="replace")
            files_in_dir = [f.name for f in dir_path.iterdir() if f.is_file()]
            files_str = "\n".join(f"  - {f}" for f in files_in_dir)
            return f"# SKILL Package: {skill_name}\n\nFiles in package:\n{files_str}\n\n## Documentation\n\n{content}"
            
    file_path = skills_dir / f"{skill_name}.md"
    if file_path.is_file():
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return f"# SKILL: {skill_name}\n\n{content}"
        
    raise ValueError(f"SKILL '{raw_name}' not found. Use list_skills to see available skills.")




_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
    "list_skills": tool_list_skills,
    "read_skill": tool_read_skill
}