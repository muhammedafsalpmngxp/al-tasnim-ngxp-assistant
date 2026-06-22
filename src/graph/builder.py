"""Graph builder.

Flow:
  START → routing_node → agent_node ──[tool_calls]──► tools_node → agent_node → END
                                     └──[no calls]─────────────────────────────────┘

routing_node  — semantic routing via table vector embeddings (runs once, before agent)
agent_node    — LLM with bound tools; uses routing hint on first step
tools_node    — LangGraph ToolNode executes the chosen tools in parallel
"""
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .nodes import make_agent_node
from .router import routing_node
from .state import AgentState

logger = logging.getLogger("orchestrator.graph")


def build_app(llm=None, tools=None, checkpointer=None):
    if tools is None:
        from ..adapters import get_tools
        tools = get_tools()
    if llm is None:
        from ..llm import get_chat_model
        llm = get_chat_model()

    # parallel_tool_calls=False: prevents Groq/Llama from emitting the native
    # function-call token format that causes a 400 from the Groq API.
    bind_kwargs = {}
    from ..config import settings
    if settings.LLM_PROVIDER == "groq":
        bind_kwargs["parallel_tool_calls"] = False
    llm_with_tools = llm.bind_tools(tools, **bind_kwargs)

    graph = StateGraph(AgentState)

    graph.add_node("router", routing_node)
    graph.add_node("agent", make_agent_node(llm_with_tools))
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "router")
    graph.add_edge("router", "agent")
    graph.add_conditional_edges("agent", tools_condition)  # → "tools" or END
    graph.add_edge("tools", "agent")

    app = graph.compile(checkpointer=checkpointer or MemorySaver())
    logger.info("Graph compiled: router → agent ⇄ tools | tools=%s", [t.name for t in tools])
    return app
