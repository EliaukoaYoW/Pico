

import subprocess
import textwrap
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000

# 
DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
IGNORED_PATH_NAMES = {".git", ".pico", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}

def now():
    return datetime.now(timezone.utc).isoformat()

def clip(text, limit = MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"

def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


class WorkspaceContext:
    """ 表示当前代码仓库和工作目录的快照状态。它负责在启动 agent 时，迅速收集一些基本的环境信息（Git 状态、核心配置等）"""
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs
    
    @classmethod
    def build(cls, cwd, repo_root_override = None):
        cwd = Path(cwd).resolve()

        def git(args, fallback = ""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd = cwd,
                    capture_output = True,
                    text = True,
                    check = True,
                    timeout = 5
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        # 确定 Git 仓库根目录。如果没有提供 override，则通过 `git rev-parse` 获取。
        repo_root = (
            Path(repo_root_override).resolve()
            if repo_root_override is not None
            else Path(git(["rev-parse","--show-toplevel"], str(cwd))).resolve()
        )
        docs = {}
        # 同时扫描 repo_root 和 cwd 下的文档 这样在子目录启动时也能看到本地文档
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)
        
        return cls(
            cwd = str(cwd),
            repo_root = str(repo_root),
            branch = git(["branch", "--show-current"],"-") or "-",  # 当前所在 Git 分支
            default_branch = (
                lambda branch: branch[len("origin/") :] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),  # 获取远端 origin 对应的默认分支（例如 origin/main 或 origin/master）
            status = clip(git(["status", "--short"], "clean") or "clean", 1500),  # 当前 Git 状态
            recent_commits = [line for line in git(["log", "--online", "5"]).splitlines() if line],  # 最近 5 个提交记录
            project_docs = docs
        )
    
    def text(self):
        # 这段文本会被塞进 Prompt Prefix 作为相对稳定的基线上下文
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path,snippet in self.project_docs.items()) or "- none"
        return textwrap.dedent(
            f"""\
            Workspace:
            - cwd: {self.cwd}
            - repo_root: {self.repo_root}
            - branch: {self.branch}
            - default_branch: {self.default_branch}
            - status:
            {self.status}
            - recent_commits:
            {commits}
            - project_docs:
            {docs}
            """
        ).strip()

    def fingerprint(self):
        """ 生成工作空间的唯一标识符，用于缓存和比较不同工作空间的状态 """
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": self.project_docs
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    ws = WorkspaceContext.build(".")
    print(ws.text())