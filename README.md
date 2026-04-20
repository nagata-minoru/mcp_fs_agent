# gemma4 MCP ファイルシステムエージェント

Ollama で動作する `gemma4:e2b` モデルに、MCP（Model Context Protocol）経由でカレントディレクトリへの読み書きアクセスを提供するインタラクティブな CLI エージェントです。

## 必要環境

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)（`gemma4:e2b` をプル済み）
- Node.js / npx

## 使い方

スクリプトを実行したいディレクトリで以下を実行します。

```bash
uv run mcp_fs_agent.py
```

初回実行時は `npx` が `@modelcontextprotocol/server-filesystem` を自動ダウンロードします。

## 操作

```
You: ここにあるファイルを一覧して
You: hello.txt というファイルを作って「こんにちは」と書いて
You: exit   # または Ctrl+C で終了
```

## 仕組み

```
ユーザー入力
    ↓
Ollama (gemma4:e2b) — ツール呼び出しを判断
    ↓
MCP filesystem server — ファイル操作を実行
    ↓
結果を Ollama に返し、最終応答を生成
```
