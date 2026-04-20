#!/usr/bin/env -S uv run --script
"""
gemma4:e2b + MCP filesystem agent
カレントディレクトリへの読み書きアクセスを提供する
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp", "ollama"]
# ///

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

MODEL = "gemma4:e2b"
CWD = os.getcwd()

def mcp_tools_to_ollama(tools) -> list[dict]:
  """MCPツール定義をOllamaのtool形式に変換する。"""
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

async def run():
  """MCPファイルシステムサーバーを起動し、Ollamaとの対話ループを実行する。"""
  server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", CWD],
  )

  async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
      await session.initialize()

      tools_result = await session.list_tools()
      tools = mcp_tools_to_ollama(tools_result.tools)
      print(f"[利用可能ツール] {[t['function']['name'] for t in tools]}")
      print(f"[対象ディレクトリ] {CWD}")
      print("終了するには 'exit' または Ctrl+C\n")

      messages: list[dict] = []

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
            tools=tools,
          )
          msg = response.message
          messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

          if not msg.tool_calls:
            print(f"Assistant: {msg.content}\n")
            break

          for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments or {}
            print(f"  [Tool] {name}({args})")
            result = await session.call_tool(name, arguments=args)
            content = "\n".join(
              c.text for c in result.content if hasattr(c, "text")
            )
            messages.append({"role": "tool", "content": content})

def main():
  """エントリーポイント。"""
  asyncio.run(run())

if __name__ == "__main__":
  main()
