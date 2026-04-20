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
              "Always use this absolute path for file operations and command execution. Never use relative paths. "
              "To create or overwrite a file, use the write_file tool. "
              "Shell redirection (>) is not supported in shell_execute; use bash -c '...' if you need it."
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

            while True:
              response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=all_tools,
              )
              msg = response.message
              messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

              if not msg.tool_calls:
                print(f"Assistant: {msg.content}\n")
                break

              for tc in msg.tool_calls:
                name = tc.function.name
                args = normalize_shell_args(name, tc.function.arguments or {})
                print(f"  [Tool] {name}({args})")
                session = tool_registry.get(name, fs_session)
                result = await session.call_tool(name, arguments=args)
                content = "\n".join(
                  c.text for c in result.content if hasattr(c, "text")
                )
                messages.append({"role": "tool", "content": content})

def main():
  asyncio.run(run())

if __name__ == "__main__":
  main()
