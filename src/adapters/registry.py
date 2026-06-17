"""Tool registry — no if/elif dispatch. Add a tool by decorating with @register_tool."""
from typing import List

from langchain_core.tools import BaseTool

_REGISTRY: List[BaseTool] = []


def register_tool(tool_obj: BaseTool) -> BaseTool:
    if tool_obj.name in {t.name for t in _REGISTRY}:
        return tool_obj
    _REGISTRY.append(tool_obj)
    return tool_obj


def get_tools() -> List[BaseTool]:
    return list(_REGISTRY)


def tool_names() -> List[str]:
    return [t.name for t in _REGISTRY]
