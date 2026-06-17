"""Graph state. add_messages APPENDS each node's messages (feeds tool results back)."""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int
