import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_fs_agent import mcp_tools_to_ollama, run, normalize_shell_args, ALLOW_COMMANDS

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

class TestAllowCommands:
  def test_デフォルトのホワイトリストにgitが含まれる(self):
    assert "git" in ALLOW_COMMANDS.split(",")

  def test_デフォルトのホワイトリストにlsが含まれる(self):
    assert "ls" in ALLOW_COMMANDS.split(",")

  def test_デフォルトのホワイトリストにbashが含まれる(self):
    assert "bash" in ALLOW_COMMANDS.split(",")

class TestNormalizeShellArgs:
  def test_スペースを含む要素を分割する(self):
    args = {"command": ["git init"], "directory": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result["command"] == ["git", "init"]

  def test_スペースなし要素はそのまま(self):
    args = {"command": ["git", "init"], "directory": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result["command"] == ["git", "init"]

  def test_shell_execute以外はそのまま(self):
    args = {"command": ["git init"]}
    result = normalize_shell_args("read_file", args)
    assert result["command"] == ["git init"]

  def test_commandキーがなければそのまま(self):
    args = {"path": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result == {"path": "/tmp"}

  def test_複数引数を含む要素を分割する(self):
    args = {"command": ["ls -la /tmp"], "directory": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result["command"] == ["ls", "-la", "/tmp"]

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
  async def test_システムプロンプトにファイル作成とbash指示が含まれる(self):
    fs_cm, _ = make_session_mock([make_mcp_tool("read_file", "読む", {})])
    sh_cm, _ = make_session_mock([make_mcp_tool("shell_execute", "実行", {})])

    mock_msg = MagicMock(content="ok", tool_calls=None)
    captured = {}

    def capture_chat(**kwargs):
      captured["messages"] = kwargs.get("messages", [])
      return MagicMock(message=mock_msg)

    with (
      patch("mcp_fs_agent.stdio_client", side_effect=[make_stdio_mock(), make_stdio_mock()]),
      patch("mcp_fs_agent.ClientSession", side_effect=[fs_cm, sh_cm]),
      patch("mcp_fs_agent.ollama.chat", side_effect=capture_chat),
      patch("builtins.input", side_effect=["hello", "exit"]),
      patch("builtins.print"),
    ):
      await run()

    system_msg = captured["messages"][0]
    assert system_msg["role"] == "system"
    assert "write_file" in system_msg["content"]
    assert "bash -c" in system_msg["content"]
    assert "NEVER output code as text" in system_msg["content"]

  @pytest.mark.asyncio
  async def test_コードブロックを返した場合にナッジする(self):
    fs_tool = make_mcp_tool("write_file", "書く", {})
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, _ = make_session_mock([sh_tool])

    mock_tool_content = MagicMock()
    mock_tool_content.text = "ok"
    mock_tool_result = MagicMock()
    mock_tool_result.content = [mock_tool_content]
    fs_session.call_tool.return_value = mock_tool_result

    mock_tc = MagicMock()
    mock_tc.function.name = "write_file"
    mock_tc.function.arguments = {"path": "/tmp/hello.py", "content": "print('hello')"}

    msg_with_code = MagicMock(content="```python\nprint('hello')\n```", tool_calls=None)
    msg_with_tool = MagicMock(content="", tool_calls=[mock_tc])
    msg_final = MagicMock(content="書きました", tool_calls=None)

    responses = iter([
      MagicMock(message=msg_with_code),
      MagicMock(message=msg_with_tool),
      MagicMock(message=msg_final),
    ])

    with (
      patch("mcp_fs_agent.stdio_client", side_effect=[make_stdio_mock(), make_stdio_mock()]),
      patch("mcp_fs_agent.ClientSession", side_effect=[fs_cm, sh_cm]),
      patch("mcp_fs_agent.ollama.chat", side_effect=responses),
      patch("builtins.input", side_effect=["hello.pyを作って", "exit"]),
      patch("builtins.print"),
    ):
      await run()

    fs_session.call_tool.assert_called_once_with("write_file", arguments={"path": "/tmp/hello.py", "content": "print('hello')"})

  @pytest.mark.asyncio
  async def test_ナッジは最大2回まで(self):
    fs_cm, _ = make_session_mock([make_mcp_tool("write_file", "書く", {})])
    sh_cm, _ = make_session_mock([make_mcp_tool("shell_execute", "実行", {})])

    msg_with_code = MagicMock(content="```python\nprint('hello')\n```", tool_calls=None)
    chat_call_count = 0

    def count_chat(**kwargs):
      nonlocal chat_call_count
      chat_call_count += 1
      return MagicMock(message=msg_with_code)

    with (
      patch("mcp_fs_agent.stdio_client", side_effect=[make_stdio_mock(), make_stdio_mock()]),
      patch("mcp_fs_agent.ClientSession", side_effect=[fs_cm, sh_cm]),
      patch("mcp_fs_agent.ollama.chat", side_effect=count_chat),
      patch("builtins.input", side_effect=["hello.pyを作って", "exit"]),
      patch("builtins.print"),
    ):
      await run()

    assert chat_call_count == 3

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
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
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
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, sh_session = make_session_mock([sh_tool])

    mock_tool_content = MagicMock()
    mock_tool_content.text = "hello"
    mock_tool_result = MagicMock()
    mock_tool_result.content = [mock_tool_content]
    sh_session.call_tool.return_value = mock_tool_result

    mock_tc = MagicMock()
    mock_tc.function.name = "shell_execute"
    mock_tc.function.arguments = {"command": ["echo hello"], "directory": "/tmp"}

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

    sh_session.call_tool.assert_called_once_with(
      "shell_execute", arguments={"command": ["echo", "hello"], "directory": "/tmp"}
    )
    fs_session.call_tool.assert_not_called()
