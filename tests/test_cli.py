import pytest

from tool_agent.cli import main


@pytest.mark.parametrize("engine", ["agent", "manual"])
def test_cli_answer(capsys, engine):
    assert main(["three times two", "-p", "offline", "-e", engine]) == 0
    assert "3 × 2 = 6" in capsys.readouterr().out


@pytest.mark.parametrize("engine", ["agent", "manual"])
def test_cli_trace(capsys, engine):
    main(["Add 3 and 2, then multiply by 4", "-p", "offline", "-e", engine, "--trace"])
    out = capsys.readouterr().out
    for expected in ["Human", "add", "ToolMessage", "multiply", "AI answer", "id=call_"]:
        assert expected in out


@pytest.mark.parametrize("engine", ["agent", "manual"])
def test_cli_interactive(monkeypatch, capsys, engine):
    answers = iter(["one plus 2", "tip on $60 at 20%", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["-p", "offline", "-e", engine]) == 0
    out = capsys.readouterr().out
    assert "1 + 2 = 3" in out and "$72.00" in out and "bye" in out
