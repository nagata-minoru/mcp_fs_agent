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
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

MODEL = "gemma4:e2b"
CWD = os.getcwd()
ALLOW_COMMANDS = os.environ.get("ALLOW_COMMANDS", "ls,cat,pwd,grep,wc,find,echo,python,uv,git,ps,kill,bash")

def mcp_tools_to_ollama(tools) -> list[dict]:
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
  for msg in reversed(messages):
    if msg.get("role") != "user":
      continue
    matches = re.findall(r'(?<![a-zA-Z0-9_\-])[a-zA-Z0-9][a-zA-Z0-9_\-]*\.[a-zA-Z][a-zA-Z0-9]*(?![a-zA-Z0-9_\-])', msg.get("content", ""))
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

async def run():
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

          messages: list[dict] = [
            {"role": "system", "content": (
              f"The working directory is {CWD}. "
              "You are an autonomous agent. Always use tools to complete tasks — never just explain or describe what you would do. "
              "Always use absolute paths. "
              "To create or overwrite a file, use the write_file tool. "
              "NEVER output code as text in your response — always write it to a file immediately using write_file. "
              "If a tool call fails, fix the arguments and retry — never ask the user to manually perform file operations. "
              "Shell redirection (>) is not supported in shell_execute; use bash -c '...' if you need it. "
              "When writing Python code, use 2-space indentation. "
              "Always respond to the user in Japanese."
            )},
          ]

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

            nudge_count = 0
            last_had_error = False
            last_error_content = ""
            while True:
              response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=all_tools,
              )
              msg = response.message
              messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

              if not msg.tool_calls:
                had_error = last_had_error
                last_had_error = False
                if nudge_count < 2 and (had_error or "```" in (msg.content or "")):
                  nudge_count += 1
                  nudge_msg = (
                    f"The previous tool call failed with: {last_error_content}. Fix the error and retry."
                    if had_error else
                    "Now write that code to a file using write_file."
                  )
                  messages.append({"role": "user", "content": nudge_msg})
                  continue
                print(f"Assistant: {msg.content}\n")
                break

              last_had_error = False
              for tc in msg.tool_calls:
                name = tc.function.name
                args = normalize_shell_args(name, tc.function.arguments or {})
                if name == "write_file" and "path" not in args:
                  filename = extract_filename_from_messages(messages)
                  args = {**args, "path": f"{CWD}/{filename}"}
                  print(f"  [path補完] {args['path']}")
                print(f"  [Tool] {name}({args})")
                session = tool_registry.get(name, fs_session)
                result = await session.call_tool(name, arguments=args)
                content = "\n".join(
                  c.text for c in result.content if hasattr(c, "text")
                )
                if result.isError:
                  print(f"  [Error] {content}")
                  last_had_error = True
                  last_error_content = content
                elif name == "write_file" and args.get("path", "").endswith(".py"):
                  try:
                    py_compile.compile(args["path"], doraise=True)
                  except py_compile.PyCompileError as e:
                    syntax_error = str(e)
                    print(f"  [SyntaxError] {syntax_error}")
                    content = f"File written but has a syntax error: {syntax_error}"
                    last_had_error = True
                    last_error_content = content
                messages.append({"role": "tool", "content": content})

def main():
  asyncio.run(run())

if __name__ == "__main__":
  main()
