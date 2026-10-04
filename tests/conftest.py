import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedToolModel(GenericFakeChatModel):
    """Fake chat model replaying scripted AIMessages (incl. tool calls); records what it was sent."""

    received: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def calls(*specs):
    """calls(("add", {"a": 1, "b": 2}), ...) -> one AIMessage with these tool calls."""
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{i}"} for i, (n, a) in enumerate(specs)])


@pytest.fixture
def scripted_model():
    def make(*messages):
        return ScriptedToolModel(messages=iter(messages), received=[])
    return make
