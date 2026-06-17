"""Graph nodes. The agent node logs each step and the routing decision."""
import logging

from langchain_core.messages import AIMessage, SystemMessage

from ..config import settings
from ..prompts import MAX_ITERATIONS_MESSAGE, SYSTEM_PROMPT
from .state import AgentState

logger = logging.getLogger("orchestrator.agent")

_SYSTEM = SystemMessage(content=SYSTEM_PROMPT)


def make_agent_node(llm_with_tools):
    def agent_node(state: AgentState) -> dict:
        n = state.get("iterations", 0)
        if n >= settings.MAX_ITERATIONS:
            logger.warning("iteration cap (%s) reached -> forcing final answer", settings.MAX_ITERATIONS)
            return {"messages": [AIMessage(content=MAX_ITERATIONS_MESSAGE)], "iterations": n + 1}

        logger.info("step %d: asking LLM to decide", n + 1)
        response = llm_with_tools.invoke([_SYSTEM] + list(state["messages"]))
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            logger.info("ROUTING -> %s", [tc.get("name") for tc in tool_calls])
        else:
            logger.info("no tool calls -> final answer ready")
        return {"messages": [response], "iterations": n + 1}

    return agent_node
