import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_fs_agent import mcp_tools_to_ollama, run, normalize_shell_args, ALLOW_COMMANDS, extract_filename_from_messages

def make_mcp_tool(name: str, description: str | None, input_schema: dict) -> MagicMock:
  """MCP ツールのモックを生成する。"""
  tool = MagicMock()
  tool.name = name
  tool.description = description
  tool.inputSchema = input_schema
  return tool

def make_stdio_mock():
  """stdio_client 用の非同期コンテキストマネージャモックを生成する。"""
  m = MagicMock()
  m.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
  m.__aexit__ = AsyncMock(return_value=False)
  return m

def make_session_mock(tools: list) -> AsyncMock:
  """ClientSession 用のモックとセッションオブジェクトを返す。"""
  tools_result = MagicMock()
  tools_result.tools = tools
  session = AsyncMock()
  session.list_tools.return_value = tools_result
  cm = AsyncMock()
  cm.__aenter__ = AsyncMock(return_value=session)
  cm.__aexit__ = AsyncMock(return_value=False)
  return cm, session

class TestAllowCommands:
  """ALLOW_COMMANDS のデフォルト値テスト。"""

  def test_デフォルトのホワイトリストにgitが含まれる(self):
    """git が ALLOW_COMMANDS のデフォルト値に含まれることを確認する。"""
    assert "git" in ALLOW_COMMANDS.split(",")

  def test_デフォルトのホワイトリストにlsが含まれる(self):
    """ls が ALLOW_COMMANDS のデフォルト値に含まれることを確認する。"""
    assert "ls" in ALLOW_COMMANDS.split(",")

  def test_デフォルトのホワイトリストにbashが含まれる(self):
    """bash が ALLOW_COMMANDS のデフォルト値に含まれることを確認する。"""
    assert "bash" in ALLOW_COMMANDS.split(",")

class TestNormalizeShellArgs:
  """normalize_shell_args() のテスト。"""

  def test_スペースを含む要素を分割する(self):
    """スペースを含む要素を shlex.split で分割することを確認する。"""
    args = {"command": ["git init"], "directory": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result["command"] == ["git", "init"]

  def test_スペースなし要素はそのまま(self):
    """スペースを含まない要素はそのまま維持されることを確認する。"""
    args = {"command": ["git", "init"], "directory": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result["command"] == ["git", "init"]

  def test_shell_execute以外はそのまま(self):
    """shell_execute 以外のツール名の場合は引数を変更しないことを確認する。"""
    args = {"command": ["git init"]}
    result = normalize_shell_args("read_file", args)
    assert result["command"] == ["git init"]

  def test_commandキーがなければそのまま(self):
    """command キーがない場合は引数をそのまま返すことを確認する。"""
    args = {"path": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result == {"path": "/tmp"}

  def test_複数引数を含む要素を分割する(self):
    """スペース区切りの複数引数を正しく分割することを確認する。"""
    args = {"command": ["ls -la /tmp"], "directory": "/tmp"}
    result = normalize_shell_args("shell_execute", args)
    assert result["command"] == ["ls", "-la", "/tmp"]

class TestExtractFilenameFromMessages:
  """extract_filename_from_messages() のテスト。"""

  def test_ユーザーメッセージからファイル名を抽出する(self):
    """user ロールのメッセージからファイル名を正規表現で抽出することを確認する。"""
    messages = [{"role": "user", "content": "tetris.py を作って"}]
    assert extract_filename_from_messages(messages) == "tetris.py"

  def test_複数ファイル名があれば最初を返す(self):
    """複数のファイル名が含まれる場合は最初に見つかったものを返すことを確認する。"""
    messages = [{"role": "user", "content": "game.py と utils.py を作って"}]
    assert extract_filename_from_messages(messages) == "game.py"

  def test_ファイル名がなければデフォルトを返す(self):
    """ファイル名パターンが見つからない場合は output.py を返すことを確認する。"""
    messages = [{"role": "user", "content": "テトリスを作って"}]
    assert extract_filename_from_messages(messages) == "output.py"

  def test_直近のユーザーメッセージを優先する(self):
    """逆順に検索するため直近のユーザーメッセージのファイル名が優先されることを確認する。"""
    messages = [
      {"role": "user", "content": "old.py を作って"},
      {"role": "assistant", "content": "ok"},
      {"role": "user", "content": "new.py に書いて"},
    ]
    assert extract_filename_from_messages(messages) == "new.py"

  def test_userロール以外は無視する(self):
    """assistant などの user 以外のロールのメッセージは検索対象外であることを確認する。"""
    messages = [
      {"role": "assistant", "content": "assistant.py"},
      {"role": "user", "content": "ファイルを作って"},
    ]
    assert extract_filename_from_messages(messages) == "output.py"

class TestMcpToolsToOllama:
  """mcp_tools_to_ollama() のテスト。"""

  def test_単一ツールを変換できる(self):
    """MCP ツールを Ollama 形式の function 定義に変換することを確認する。"""
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
    """ツールリストが空の場合は空リストを返すことを確認する。"""
    assert mcp_tools_to_ollama([]) == []

  def test_descriptionがNoneの場合は空文字になる(self):
    """description が None の場合は空文字列に変換されることを確認する。"""
    tools = [make_mcp_tool("list_dir", None, {})]
    result = mcp_tools_to_ollama(tools)
    assert result[0]["function"]["description"] == ""

  def test_複数ツールを変換できる(self):
    """複数のツールが順序を保って変換されることを確認する。"""
    tools = [
      make_mcp_tool("read_file", "読む", {}),
      make_mcp_tool("write_file", "書く", {}),
    ]
    result = mcp_tools_to_ollama(tools)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "read_file"
    assert result[1]["function"]["name"] == "write_file"

  def test_inputSchemaがそのまま渡される(self):
    """inputSchema オブジェクトが参照渡しされることを確認する。"""
    schema = {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}
    tools = [make_mcp_tool("read_file", "", schema)]
    result = mcp_tools_to_ollama(tools)
    assert result[0]["function"]["parameters"] is schema

class TestRun:
  """run() の統合テスト。"""

  @pytest.mark.asyncio
  async def test_システムプロンプトにファイル作成とbash指示が含まれる(self):
    """システムプロンプトに write_file・bash -c・コードテキスト出力禁止の指示が含まれることを確認する。"""
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
    assert "Japanese" in system_msg["content"]

  @pytest.mark.asyncio
  async def test_コードブロックを返した場合にナッジする(self):
    """モデルがコードブロックを含む返答をしてツールを呼ばなかった場合にナッジを送ることを確認する。"""
    fs_tool = make_mcp_tool("write_file", "書く", {})
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, _ = make_session_mock([sh_tool])

    mock_tool_content = MagicMock()
    mock_tool_content.text = "ok"
    mock_tool_result = MagicMock()
    mock_tool_result.isError = False
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

    fs_session.call_tool.assert_called_once_with(
      "write_file", arguments={"path": "/tmp/hello.py", "content": "print('hello')"}
    )

  @pytest.mark.asyncio
  async def test_ナッジは最大2回まで(self):
    """コードブロックが返り続けてもナッジは最大2回で打ち切られることを確認する。"""
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
  async def test_write_fileにpathがなければメッセージから補完する(self):
    """write_file に path が渡されなかった場合、ユーザーメッセージからファイル名を推定して補完することを確認する。"""
    fs_tool = make_mcp_tool("write_file", "書く", {})
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, _ = make_session_mock([sh_tool])

    mock_tool_result = MagicMock()
    mock_tool_result.isError = False
    mock_tool_result.content = [MagicMock(text="ok")]
    fs_session.call_tool.return_value = mock_tool_result

    mock_tc = MagicMock()
    mock_tc.function.name = "write_file"
    mock_tc.function.arguments = {"content": "print('hello')"}

    msg_with_tool = MagicMock(content="", tool_calls=[mock_tc])
    msg_final = MagicMock(content="書きました", tool_calls=None)

    responses = iter([
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

    call_args = fs_session.call_tool.call_args
    assert call_args[1]["arguments"]["content"] == "print('hello')"
    assert call_args[1]["arguments"]["path"].endswith("hello.py")

  @pytest.mark.asyncio
  async def test_ツールエラー後にリトライナッジを送る(self):
    """ツール呼び出しがエラーになった後にモデルがツールを呼ばない返答をした場合にリトライナッジを送ることを確認する。"""
    fs_tool = make_mcp_tool("write_file", "書く", {})
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, _ = make_session_mock([sh_tool])

    error_result = MagicMock()
    error_result.isError = True
    error_result.content = [MagicMock(text="Invalid input: expected string, received undefined")]

    success_result = MagicMock()
    success_result.isError = False
    success_result.content = [MagicMock(text="ok")]

    fs_session.call_tool.side_effect = [error_result, success_result]

    mock_tc_bad = MagicMock()
    mock_tc_bad.function.name = "write_file"
    mock_tc_bad.function.arguments = {}

    mock_tc_good = MagicMock()
    mock_tc_good.function.name = "write_file"
    mock_tc_good.function.arguments = {"path": "/tmp/hello.py", "content": "print('hello')"}

    msg_with_bad_tool = MagicMock(content="", tool_calls=[mock_tc_bad])
    msg_empty = MagicMock(content="", tool_calls=None)
    msg_with_good_tool = MagicMock(content="", tool_calls=[mock_tc_good])
    msg_final = MagicMock(content="書きました", tool_calls=None)

    responses = iter([
      MagicMock(message=msg_with_bad_tool),
      MagicMock(message=msg_empty),
      MagicMock(message=msg_with_good_tool),
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

    assert fs_session.call_tool.call_count == 2

  @pytest.mark.asyncio
  async def test_リトライナッジにエラー内容が含まれる(self):
    """リトライナッジのメッセージに実際のエラーメッセージが含まれることを確認する。"""
    fs_tool = make_mcp_tool("write_file", "書く", {})
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, _ = make_session_mock([sh_tool])

    error_result = MagicMock()
    error_result.isError = True
    error_result.content = [MagicMock(text="Invalid input: expected string, received undefined")]
    fs_session.call_tool.return_value = error_result

    mock_tc = MagicMock()
    mock_tc.function.name = "write_file"
    mock_tc.function.arguments = {}

    captured_messages = []

    def capture_chat(**kwargs):
      captured_messages.extend(kwargs.get("messages", []))
      msg_empty = MagicMock(content="", tool_calls=None)
      return MagicMock(message=msg_empty)

    msg_with_tool = MagicMock(content="", tool_calls=[mock_tc])
    first_response = MagicMock(message=msg_with_tool)

    call_count = 0
    def side_effect_chat(**kwargs):
      nonlocal call_count
      call_count += 1
      captured_messages.clear()
      captured_messages.extend(kwargs.get("messages", []))
      if call_count == 1:
        return first_response
      msg_empty = MagicMock(content="", tool_calls=None)
      return MagicMock(message=msg_empty)

    with (
      patch("mcp_fs_agent.stdio_client", side_effect=[make_stdio_mock(), make_stdio_mock()]),
      patch("mcp_fs_agent.ClientSession", side_effect=[fs_cm, sh_cm]),
      patch("mcp_fs_agent.ollama.chat", side_effect=side_effect_chat),
      patch("builtins.input", side_effect=["hello.pyを作って", "exit"]),
      patch("builtins.print"),
    ):
      await run()

    nudge_msgs = [m for m in captured_messages if m.get("role") == "user" and "failed with" in m.get("content", "")]
    assert len(nudge_msgs) >= 1
    assert "Invalid input: expected string, received undefined" in nudge_msgs[0]["content"]

  @pytest.mark.asyncio
  async def test_exitで終了する(self):
    """exit と入力した場合にエラーなくループを抜けて終了することを確認する。"""
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
    """read_file などのファイルシステムツールは fs セッション経由で実行されることを確認する。"""
    fs_tool = make_mcp_tool("read_file", "読む", {})
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, sh_session = make_session_mock([sh_tool])

    mock_tool_content = MagicMock()
    mock_tool_content.text = "ファイルの内容"
    mock_tool_result = MagicMock()
    mock_tool_result.isError = False
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
    """shell_execute などのシェルツールは sh セッション経由で実行されることを確認する。"""
    fs_tool = make_mcp_tool("read_file", "読む", {})
    sh_tool = make_mcp_tool("shell_execute", "実行", {})
    fs_cm, fs_session = make_session_mock([fs_tool])
    sh_cm, sh_session = make_session_mock([sh_tool])

    mock_tool_content = MagicMock()
    mock_tool_content.text = "hello"
    mock_tool_result = MagicMock()
    mock_tool_result.isError = False
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
