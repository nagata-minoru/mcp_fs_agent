import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_fs_agent import mcp_tools_to_ollama, run

def make_mcp_tool(name: str, description: str | None, input_schema: dict) -> MagicMock:
  """テスト用のMCPツールモックを生成する。"""
  tool = MagicMock()
  tool.name = name
  tool.description = description
  tool.inputSchema = input_schema
  return tool

class TestMcpToolsToOllama:
  """mcp_tools_to_ollama のユニットテスト。"""

  def test_単一ツールを変換できる(self):
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    tools = [make_mcp_tool("read_file", "ファイルを読む", schema)]
    result = mcp_tools_to_ollama(tools)
    assert len(result) == 1
    f = result[0]["function"]
    assert result[0]["type"] == "function"
    assert f["name"] == "read_file"
    assert f["description"] == "ファイルを読む"
    assert f["parameters"] == schema

  def test_空リストを渡すと空リストを返す(self):
    assert mcp_tools_to_ollama([]) == []

  def test_descriptionがNoneの場合は空文字になる(self):
    tools = [make_mcp_tool("list_dir", None, {})]
    result = mcp_tools_to_ollama(tools)
    assert result[0]["function"]["description"] == ""

  def test_複数ツールを変換できる(self):
    tools = [
      make_mcp_tool("read_file", "読む", {}),
      make_mcp_tool("write_file", "書く", {}),
    ]
    result = mcp_tools_to_ollama(tools)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "read_file"
    assert result[1]["function"]["name"] == "write_file"

  def test_inputSchemaがそのまま渡される(self):
    schema = {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}
    tools = [make_mcp_tool("read_file", "", schema)]
    result = mcp_tools_to_ollama(tools)
    assert result[0]["function"]["parameters"] is schema

class TestRun:
  """run() の統合テスト（外部依存はモック）。"""

  @pytest.mark.asyncio
  async def test_exitで終了する(self):
    """'exit' 入力でループを抜けることを確認する。"""
    mcp_tool = make_mcp_tool("read_file", "読む", {})

    mock_tools_result = MagicMock()
    mock_tools_result.tools = [mcp_tool]

    mock_session = AsyncMock()
    mock_session.list_tools.return_value = mock_tools_result

    mock_msg = MagicMock()
    mock_msg.content = "こんにちは"
    mock_msg.tool_calls = None
    mock_response = MagicMock()
    mock_response.message = mock_msg

    with (
      patch("mcp_fs_agent.stdio_client") as mock_stdio,
      patch("mcp_fs_agent.ollama.chat", return_value=mock_response),
      patch("builtins.input", side_effect=["exit"]),
      patch("builtins.print"),
    ):
      mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
      mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

      mock_cm = AsyncMock()
      mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
      mock_cm.__aexit__ = AsyncMock(return_value=False)

      with patch("mcp_fs_agent.ClientSession", return_value=mock_cm):
        await run()

  @pytest.mark.asyncio
  async def test_ツール呼び出しを実行しモデルに結果を返す(self):
    """モデルがツール呼び出しを返した場合にMCPを呼び出すことを確認する。"""
    mcp_tool = make_mcp_tool("read_file", "読む", {})

    mock_tools_result = MagicMock()
    mock_tools_result.tools = [mcp_tool]

    mock_tool_content = MagicMock()
    mock_tool_content.text = "ファイルの内容"
    mock_tool_result = MagicMock()
    mock_tool_result.content = [mock_tool_content]

    mock_session = AsyncMock()
    mock_session.list_tools.return_value = mock_tools_result
    mock_session.call_tool.return_value = mock_tool_result

    mock_tc = MagicMock()
    mock_tc.function.name = "read_file"
    mock_tc.function.arguments = {"path": "hello.txt"}

    msg_with_tool = MagicMock()
    msg_with_tool.content = ""
    msg_with_tool.tool_calls = [mock_tc]

    msg_final = MagicMock()
    msg_final.content = "読みました"
    msg_final.tool_calls = None

    responses = iter([
      MagicMock(message=msg_with_tool),
      MagicMock(message=msg_final),
    ])

    with (
      patch("mcp_fs_agent.stdio_client") as mock_stdio,
      patch("mcp_fs_agent.ollama.chat", side_effect=responses),
      patch("builtins.input", side_effect=["hello.txtを読んで", "exit"]),
      patch("builtins.print"),
    ):
      mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
      mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

      mock_cm = AsyncMock()
      mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
      mock_cm.__aexit__ = AsyncMock(return_value=False)

      with patch("mcp_fs_agent.ClientSession", return_value=mock_cm):
        await run()

    mock_session.call_tool.assert_called_once_with("read_file", arguments={"path": "hello.txt"})
