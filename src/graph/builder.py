"""Graph builder: START -> agent -> (tools_condition) -> tools -> agent -> ... -> END."""
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .nodes import make_agent_node
from .state import AgentState

logger = logging.getLogger("orchestrator.graph")


def build_app(llm=None, tools=None, checkpointer=None):
    if tools is None:
        from ..adapters import get_tools  # triggers tool registration
        tools = get_tools()
    if llm is None:
        from ..llm import get_chat_model
        llm = get_chat_model()

    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm_with_tools))
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    graph.add_edge("tools", "agent")

    app = graph.compile(checkpointer=checkpointer or MemorySaver())  # compiled ONCE
    logger.info("Graph compiled with tools: %s", [t.name for t in tools])
    return app
