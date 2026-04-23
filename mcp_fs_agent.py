#!/usr/bin/env -S uv run --script
"""
gemma4:e2b + MCP filesystem + shell agent
カレントディレクトリへの読み書きとコマンド実行を提供する
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp", "ollama"]
# ///

import asyncio
import os
import py_compile
import re
import readline  # noqa: F401
import shlex
import subprocess
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

MODEL = "gemma4:e2b"
_FILENAME_RE = re.compile(r'(?<![a-zA-Z0-9_\-])[a-zA-Z0-9][a-zA-Z0-9_\-]*\.[a-zA-Z][a-zA-Z0-9]*(?![a-zA-Z0-9_\-])')
CWD = os.getcwd()
ALLOW_COMMANDS = os.environ.get("ALLOW_COMMANDS", "ls,cat,pwd,grep,wc,find,echo,python,uv,git,ps,kill,bash")
_COMMIT_REQUEST_RE = re.compile(r"(コミット|commit)", re.IGNORECASE)
_COMMIT_MESSAGE_ONLY_RE = re.compile(r"(コミットメッセージ|commit message)", re.IGNORECASE)
_ANSI_RE = re.compile(r'\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|[^[\]])')
_GENERIC_RESPONSE_MARKERS = (
  "何かご依頼",
  "お答えします",
  "ご質問",
  "お手伝い",
)
_TEXT_TOOL_CALL_RE = re.compile(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(")

SYSTEM_PROMPT = (
  f"The working directory is {CWD}. "
  "NEVER ask the user any questions — if you need information, use tools to get it yourself. "
  "You are an autonomous agent. Always use tools to complete tasks — never just explain or describe what you would do. "
  "NEVER run git commands unless the user explicitly uses the word 'コミット' or 'commit'. "
  "When asked to commit: call shell_execute(['git', 'status', '--short']), shell_execute(['git', 'diff']), "
  "then shell_execute(['git', 'add', '-A']), then shell_execute(['git', 'commit', '-m', '<Japanese message>']). "
  "If there are no changes, reply: 「コミットする変更がありません。」. "
  "Always use absolute paths. "
  "To create or overwrite a file, use the write_file tool. "
  "NEVER output code as text in your response — always write it to a file immediately using write_file. "
  "If a tool call fails, fix the arguments and retry — never ask the user to manually perform file operations. "
  "Shell redirection (>) is not supported in shell_execute; use bash -c '...' if you need it. "
  "When writing Python code, use 2-space indentation. "
  "Always respond to the user in Japanese."
)

def mcp_tools_to_ollama(tools) -> list[dict]:
  """MCP ツールリストを Ollama の function 定義形式に変換する。"""
  return [
    {
      "type": "function",
      "function": {
        "name": t.name,
        "description": t.description or "",
        "parameters": t.inputSchema,
      },
    }
    for t in tools
  ]

def extract_filename_from_messages(messages: list[dict]) -> str:
  """直近のユーザーメッセージからファイル名を推定する。見つからなければ output.py を返す。"""
  for msg in reversed(messages):
    if msg.get("role") != "user":
      continue
    matches = _FILENAME_RE.findall(msg.get("content", ""))
    if matches:
      return matches[0]
  return "output.py"

def normalize_shell_args(name: str, args: dict) -> dict:
  """モデルが command を1要素の文字列で渡した場合に split する。"""
  if name != "shell_execute" or "command" not in args:
    return args
  cmd = args["command"]
  if not isinstance(cmd, list):
    return args
  normalized = []
  for part in cmd:
    normalized.extend(shlex.split(part) if " " in part else [part])
  return {**args, "command": normalized}

def is_commit_message_request(messages: list[dict]) -> bool:
  """直近のユーザーメッセージがコミット関連依頼か判定する。"""
  for msg in reversed(messages):
    if msg.get("role") == "user":
      return bool(_COMMIT_REQUEST_RE.search(msg.get("content", "")))
  return False

def is_commit_execution_request(messages: list[dict]) -> bool:
  """コミット実行依頼か判定する（メッセージ提案のみの依頼を除外する）。"""
  for msg in reversed(messages):
    if msg.get("role") == "user":
      content = msg.get("content", "")
      if not _COMMIT_REQUEST_RE.search(content):
        return False
      return not _COMMIT_MESSAGE_ONLY_RE.search(content)
  return False

def strip_ansi(text: str) -> str:
  """ANSI エスケープシーケンスとターミナル制御コードを除去する。"""
  return _ANSI_RE.sub('', text)

def perform_git_commit(message: str, cwd: str) -> str:
  """git add -A && git commit を実行してコミット結果を返す。"""
  add = subprocess.run(['git', 'add', '-A'], capture_output=True, text=True, cwd=cwd)
  if add.returncode != 0:
    return f"git add 失敗: {add.stderr.strip()}"
  commit = subprocess.run(
    ['git', 'commit', '-m', message],
    capture_output=True, text=True, cwd=cwd
  )
  if commit.returncode != 0:
    return f"git commit 失敗: {commit.stderr.strip()}"
  return commit.stdout.strip()

def format_tool_message(name: str, args: dict, content: str) -> str:
  """モデルが次ターンで参照しやすいようにツール結果を明示的に整形する。"""
  return f"[Tool result: {name}({args})]\n{content}"

def tool_call_command(args: dict[str, Any]) -> tuple[str, ...]:
  """shell_execute の command 引数を比較しやすいタプルへ変換する。"""
  command = args.get("command", [])
  if isinstance(command, str):
    return tuple(shlex.split(command))
  if isinstance(command, list):
    return tuple(str(part) for part in command)
  return ()

def extract_git_outputs(tool_results: list[dict]) -> dict[tuple[str, ...], str]:
  """このターンで実行した git コマンドの出力を command ごとに取り出す。"""
  outputs = {}
  for result in tool_results:
    if result.get("name") != "shell_execute":
      continue
    command = tool_call_command(result.get("args", {}))
    if command and command[0] == "git":
      outputs[command] = result.get("content", "")
  return outputs

def summarize_changed_paths(status_output: str) -> list[str]:
  """git status --short の出力から変更パスを抽出する。"""
  paths = []
  for line in status_output.splitlines():
    if not line.strip():
      continue
    path = line[3:].strip() if len(line) > 3 else line.strip()
    if " -> " in path:
      path = path.split(" -> ", 1)[1]
    paths.append(path)
  return paths

def _has_git_changes(outputs: dict) -> bool:
  """git 出力から実際にコミットできる変更があるか判定する。"""
  diff = outputs.get(("git", "diff"), "")
  cached = outputs.get(("git", "diff", "--cached"), "")
  if diff.strip() or cached.strip():
    return True
  status = (outputs.get(("git", "status", "--short"), "")
            or outputs.get(("git", "status"), ""))
  return bool(status.strip()) and "nothing to commit, working tree clean" not in status

def synthesize_commit_message(tool_results: list[dict]) -> str | None:
  """git 出力から最低限妥当な日本語コミットメッセージを生成する。"""
  outputs = extract_git_outputs(tool_results)
  status_output = (outputs.get(("git", "status", "--short"), "")
                   or outputs.get(("git", "status"), ""))
  diff_output = outputs.get(("git", "diff"), "")
  cached_diff_output = outputs.get(("git", "diff", "--cached"), "")
  combined_diff = f"{diff_output}\n{cached_diff_output}"

  if not _has_git_changes(outputs):
    return "コミットする変更がありません。"

  if "mcp_fs_agent.py" in combined_diff and any(
    marker in combined_diff for marker in ("tool", "Tool", "ツール", "role")
  ):
    message = "MCPツール結果の処理を改善"
  elif "README" in combined_diff:
    message = "READMEを更新"
  elif "test_" in combined_diff or "pytest" in combined_diff:
    message = "テストを更新"
  else:
    paths = summarize_changed_paths(status_output)
    if len(paths) == 1:
      message = f"{paths[0]}を更新"
    elif paths:
      message = "関連ファイルを更新"
    else:
      message = "変更内容を更新"

  return f"コミットメッセージの提案: {message}"

def should_replace_commit_response(content: str, tool_results: list[dict]) -> bool:
  """モデルが git 出力後に汎用応答を返した場合だけローカル生成へ切り替える。"""
  if not extract_git_outputs(tool_results):
    return False
  stripped = content.strip()
  if not stripped:
    return True
  if stripped.startswith("コミットメッセージの提案:"):
    return False
  if stripped == "コミットする変更がありません。":
    return _has_git_changes(extract_git_outputs(tool_results))
  return (
    any(marker in stripped for marker in _GENERIC_RESPONSE_MARKERS)
    or bool(_TEXT_TOOL_CALL_RE.match(stripped))
  )

def looks_like_text_tool_call(content: str, tool_names: set[str]) -> bool:
  """モデルがツール呼び出しを通常テキストとして出したか判定する。"""
  stripped = content.strip()
  if not stripped:
    return False
  return any(re.match(rf"^{re.escape(name)}\s*\(", stripped) for name in tool_names)

async def execute_tool_call(
  tc,
  tool_registry: dict[str, ClientSession],
  messages: list[dict],
  default_session: ClientSession,
) -> tuple[str, bool, str]:
  """ツール呼び出しを実行し (content, had_error, error_content) を返す。"""
  name = tc.function.name
  args = normalize_shell_args(name, tc.function.arguments or {})
  if name == "write_file" and "path" not in args:
    filename = extract_filename_from_messages(messages)
    args = {**args, "path": f"{CWD}/{filename}"}
    print(f"  [path補完] {args['path']}")
  print(f"  [Tool] {name}({args})")
  session = tool_registry.get(name, default_session)
  result = await session.call_tool(name, arguments=args)
  content = strip_ansi("\n".join(c.text for c in result.content if hasattr(c, "text")))
  if result.isError:
    print(f"  [Error] {content}")
    return content, True, content
  if name == "write_file" and args.get("path", "").endswith(".py"):
    try:
      py_compile.compile(args["path"], doraise=True)
    except py_compile.PyCompileError as e:
      syntax_error = str(e)
      print(f"  [SyntaxError] {syntax_error}")
      content = f"File written but has a syntax error: {syntax_error}"
      return content, True, content
  return content, False, ""

async def agent_turn(
  messages: list[dict],
  all_tools: list[dict],
  tool_registry: dict[str, ClientSession],
  default_session: ClientSession,
) -> None:
  """1ターンのエージェントループ。ツール呼び出しが完了するまで繰り返す。"""
  nudge_count = 0
  last_had_error = False
  last_error_content = ""
  turn_tool_results: list[dict] = []
  tool_names = set(tool_registry)
  while True:
    response = ollama.chat(model=MODEL, messages=messages, tools=all_tools)
    msg = response.message
    messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

    if not msg.tool_calls:
      if is_commit_message_request(messages) and should_replace_commit_response(msg.content or "", turn_tool_results):
        fallback = synthesize_commit_message(turn_tool_results)
        if fallback:
          if (fallback != "コミットする変更がありません。"
              and is_commit_execution_request(messages)):
            commit_msg = fallback.removeprefix("コミットメッセージの提案: ")
            result = perform_git_commit(commit_msg, CWD)
            print(f"Assistant: コミットしました。\n{result}\n")
          else:
            print(f"Assistant: {fallback}\n")
          break
      had_error = last_had_error
      last_had_error = False
      is_empty = not (msg.content or "").strip()
      is_text_tool_call = looks_like_text_tool_call(msg.content or "", tool_names)
      _READ_TOOLS = {"read_text_file", "read_file", "read_multiple_files"}
      _WRITE_TOOLS = {"write_file", "edit_file"}
      read_without_write = (
        any(r["name"] in _READ_TOOLS for r in turn_tool_results)
        and not any(r["name"] in _WRITE_TOOLS for r in turn_tool_results)
        and bool(re.search(r"承知|了解|かしこまり", msg.content or ""))
      )
      if nudge_count < 2 and (
        had_error or "```" in (msg.content or "")
        or is_empty or is_text_tool_call or read_without_write
      ):
        nudge_count += 1
        if had_error:
          nudge_msg = f"The previous tool call failed with: {last_error_content}. Fix the error and retry."
        elif is_empty:
          nudge_msg = "Your response was empty. Do NOT call any tools. Write your answer as text now."
        elif is_text_tool_call:
          nudge_msg = (
            "You wrote a tool call as plain text. If you need that tool, call it through the tool API. "
            "If the task is complete, write the final Japanese answer as text now."
          )
        elif read_without_write:
          nudge_msg = (
            "You read the file but did not modify it. "
            "Fix the issue now and save the corrected file using write_file."
          )
        else:
          nudge_msg = "Now write that code to a file using write_file."
        messages.append({"role": "user", "content": nudge_msg})
        continue
      print(f"Assistant: {msg.content}\n")
      break

    last_had_error = False
    for tc in msg.tool_calls:
      content, had_error, error_content = await execute_tool_call(
        tc, tool_registry, messages, default_session
      )
      name = tc.function.name
      args = normalize_shell_args(name, tc.function.arguments or {})
      turn_tool_results.append({"name": name, "args": args, "content": content})
      if had_error:
        last_had_error = True
        last_error_content = error_content
      messages.append({"role": "tool", "name": name, "content": format_tool_message(name, args, content)})

async def run():
  """MCP サーバーに接続し、対話ループを起動する。"""
  fs_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", CWD],
  )
  shell_params = StdioServerParameters(
    command="uvx",
    args=["mcp-shell-server"],
    env={**os.environ, "ALLOW_COMMANDS": ALLOW_COMMANDS},
    cwd=CWD,
  )

  async with stdio_client(fs_params) as (fs_read, fs_write):
    async with ClientSession(fs_read, fs_write) as fs_session:
      async with stdio_client(shell_params) as (sh_read, sh_write):
        async with ClientSession(sh_read, sh_write) as sh_session:
          await fs_session.initialize()
          await sh_session.initialize()

          fs_tools = (await fs_session.list_tools()).tools
          sh_tools = (await sh_session.list_tools()).tools

          tool_registry: dict[str, ClientSession] = {}
          for t in fs_tools:
            tool_registry[t.name] = fs_session
          for t in sh_tools:
            tool_registry[t.name] = sh_session

          all_tools = mcp_tools_to_ollama(fs_tools + sh_tools)
          print(f"[利用可能ツール] {list(tool_registry.keys())}")
          print(f"[対象ディレクトリ] {CWD}")
          print(f"[許可コマンド] {ALLOW_COMMANDS}")
          print("終了するには 'exit' または Ctrl+C\n")

          messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

          while True:
            try:
              user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
              print("\n終了します。")
              break

            if user_input.lower() in ("exit", "quit", "終了"):
              break
            if not user_input:
              continue

            messages.append({"role": "user", "content": user_input})
            await agent_turn(messages, all_tools, tool_registry, fs_session)

def main():
  """エントリーポイント。"""
  asyncio.run(run())

if __name__ == "__main__":
  main()
