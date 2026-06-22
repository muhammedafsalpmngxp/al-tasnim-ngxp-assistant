"""Graph nodes. The agent node logs each step and the routing decision."""
import logging

from langchain_core.messages import AIMessage, SystemMessage

from ..config import settings
from ..prompts import MAX_ITERATIONS_MESSAGE, SYSTEM_PROMPT
from .state import AgentState

logger = logging.getLogger("orchestrator.agent")

_HINT_SUFFIX = {
    "db": (
        "\n\n[ROUTING: Semantic analysis matched this question to database tables. "
        "Call query_database.]"
    ),
    "rag": (
        "\n\n[ROUTING: No matching database tables found for this question. "
        "Call search_documents.]"
    ),
    "both": (
        "\n\n[ROUTING: This question needs both live data and document content. "
        "Call query_database AND search_documents.]"
    ),
}


def make_agent_node(llm_with_tools):
    def agent_node(state: AgentState) -> dict:
        n = state.get("iterations", 0)
        if n >= settings.MAX_ITERATIONS:
            logger.warning("iteration cap (%s) reached -> forcing final answer", settings.MAX_ITERATIONS)
            return {"messages": [AIMessage(content=MAX_ITERATIONS_MESSAGE)], "iterations": n + 1}

        # On the first step, append the semantic routing hint to the system prompt.
        # On subsequent steps the LLM already has tool results — no hint needed.
        hint = state.get("routing_hint", "")
        if n == 0 and hint in _HINT_SUFFIX:
            system_content = SYSTEM_PROMPT + _HINT_SUFFIX[hint]
        else:
            system_content = SYSTEM_PROMPT

        logger.info("step %d: asking LLM to decide (routing_hint=%r)", n + 1, hint)
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_content)] + list(state["messages"])
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            logger.info("ROUTING -> %s", [tc.get("name") for tc in tool_calls])
        else:
            logger.info("no tool calls -> final answer ready")

        return {"messages": [response], "iterations": n + 1}

    return agent_node
