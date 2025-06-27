import operator
from typing import Annotated, TypedDict, List

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sender: str