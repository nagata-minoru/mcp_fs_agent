import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_fs_agent import mcp_tools_to_ollama, run

def make_mcp_tool(name: str, description: str | None, input_schema: dict) -> MagicMock:
  tool = MagicMock()
  tool.name = name
  tool.description = description
  tool.inputSchema = input_schema
  return tool

def make_stdio_mock():
  m = MagicMock()
  m.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
  m.__aexit__ = AsyncMock(return_value=False)
  return m

def make_session_mock(tools: list) -> AsyncMock:
  tools_result = MagicMock()
  tools_result.tools = tools
  session = AsyncMock()
  session.list_tools.return_value = tools_result
  cm = AsyncMock()
  cm.__aenter__ = AsyncMock(return_value=session)
  cm.__aexit__ = AsyncMock(return_value=False)
  return cm, session

class TestMcpToolsToOllama:
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
  @pytest.mark.asyncio
  async def test_exitで終了する(self):
    fs_cm, _ = make_session_mock([make_mcp_tool("read_file", "読む", {})])
    sh_cm, _ = make_session_mock([make_mcp_tool("execute_command", "実行", {})])

    mock_msg = MagicMock()
    mock_msg.content = "こんにちは"
    mock_msg.tool_calls = None

    with (
      patch("mcp_fs_agent.stdio_client", side_effect=[make_stdio_mock(), make_stdio_mock()]),
      patch("mcp_fs_agent.ClientSession", side_effect=[fs_cm, sh_cm]),
      patch("mcp_fs_agent.ollama.chat", return_value=MagicMock(message=mock_msg)),
      patch("builtins.input", side_effect=["exit"]),
      patch("builtins.print"),
    ):
      await run()

  @pytest.mark.asyncio
  async def test_ファイルシステムツールはfsセッションで実行される(self):
    fs_tool = make_mcp_tool("read_file", "読む", {})
    sh_tool = make_mcp_tool("execute_command", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, sh_session = make_session_mock([sh_tool])

    mock_tool_content = MagicMock()
    mock_tool_content.text = "ファイルの内容"
    mock_tool_result = MagicMock()
    mock_tool_result.content = [mock_tool_content]
    fs_session.call_tool.return_value = mock_tool_result

    mock_tc = MagicMock()
    mock_tc.function.name = "read_file"
    mock_tc.function.arguments = {"path": "hello.txt"}

    msg_with_tool = MagicMock(content="", tool_calls=[mock_tc])
    msg_final = MagicMock(content="読みました", tool_calls=None)

    responses = iter([
      MagicMock(message=msg_with_tool),
      MagicMock(message=msg_final),
    ])

    with (
      patch("mcp_fs_agent.stdio_client", side_effect=[make_stdio_mock(), make_stdio_mock()]),
      patch("mcp_fs_agent.ClientSession", side_effect=[fs_cm, sh_cm]),
      patch("mcp_fs_agent.ollama.chat", side_effect=responses),
      patch("builtins.input", side_effect=["hello.txtを読んで", "exit"]),
      patch("builtins.print"),
    ):
      await run()

    fs_session.call_tool.assert_called_once_with("read_file", arguments={"path": "hello.txt"})
    sh_session.call_tool.assert_not_called()

  @pytest.mark.asyncio
  async def test_シェルツールはshセッションで実行される(self):
    fs_tool = make_mcp_tool("read_file", "読む", {})
    sh_tool = make_mcp_tool("execute_command", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, sh_session = make_session_mock([sh_tool])

    mock_tool_content = MagicMock()
    mock_tool_content.text = "hello"
    mock_tool_result = MagicMock()
    mock_tool_result.content = [mock_tool_content]
    sh_session.call_tool.return_value = mock_tool_result

    mock_tc = MagicMock()
    mock_tc.function.name = "execute_command"
    mock_tc.function.arguments = {"command": "echo hello"}

    msg_with_tool = MagicMock(content="", tool_calls=[mock_tc])
    msg_final = MagicMock(content="実行しました", tool_calls=None)

    responses = iter([
      MagicMock(message=msg_with_tool),
      MagicMock(message=msg_final),
    ])

    with (
      patch("mcp_fs_agent.stdio_client", side_effect=[make_stdio_mock(), make_stdio_mock()]),
      patch("mcp_fs_agent.ClientSession", side_effect=[fs_cm, sh_cm]),
      patch("mcp_fs_agent.ollama.chat", side_effect=responses),
      patch("builtins.input", side_effect=["echoを実行して", "exit"]),
      patch("builtins.print"),
    ):
      await run()

    sh_session.call_tool.assert_called_once_with("execute_command", arguments={"command": "echo hello"})
    fs_session.call_tool.assert_not_called()
