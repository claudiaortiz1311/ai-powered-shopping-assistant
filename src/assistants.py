# src/assistants.py
import os
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .prompts import sales_rep_prompt, support_prompt
from .state import State
from .tools import (
    DEFAULT_USER_ID,
    EscalateToHuman,
    RouteToCustomerSupport,
    cart_tool,
    search_tool,
    set_thread_id,
    set_user_id,
    structured_search_tool,
    view_cart,
)

load_dotenv()
import pandas as pd

# Setup LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "Empty"
llm = ChatOpenAI(model = 'gpt-4o', api_key=OPENAI_API_KEY)

# Tool registration
sales_tools = [
    RouteToCustomerSupport,
    search_tool,
    structured_search_tool,
    cart_tool,
    view_cart,
]
support_tools = [EscalateToHuman]

# Runnable pipelines
sales_runnable = sales_rep_prompt.partial(time=datetime.now) | llm.bind_tools(
    sales_tools
)
support_runnable = support_prompt.partial(time=datetime.now) | llm.bind_tools(
    support_tools
)

# Synchronous wrappers that call the async pipeline under the hood.
# These allow callers that use .invoke(...) (como Streamlit) to work.
sales_runnable_sync = RunnableLambda(lambda state, config=None: asyncio.run(sales_runnable.ainvoke(state, config=config)))
support_runnable_sync = RunnableLambda(lambda state, config=None: asyncio.run(support_runnable.ainvoke(state, config=config)))


async def sales_assistant(state: State, config: RunnableConfig, runnable=sales_runnable) -> dict:
    """
    LangGraph node function for running the sales assistant LLM agent.

    This function binds a chat prompt (`sales_rep_prompt`) with tools and invokes
    the LangChain Runnable pipeline. It sets the thread and user IDs and runs the
    agent with the given state and config.

    ---
    Arguments:
    - state (State): LangGraph state with current dialog history.
    - config (RunnableConfig): Config object that contains the `thread_id`.
    - runnable: (optional) The runnable to use; defaults to global `sales_runnable`.

    ---
    Behavior:
    - Extract thread ID from config and set it using `set_thread_id(...)`.
    - Set default user ID via `set_user_id(...)`.
    - Use the given `runnable` to run the assistant logic.

    ---
    Returns:
    - A dictionary with a `"messages"` key containing the new AI message(s).
    Example: `{"messages": [AIMessage(...)]}`
    """
    # extract thread_id robustly
    thread_id = None
    try:
        thread_id = config["configurable"]["thread_id"]
    except Exception:
        try:
            thread_id = getattr(config, "configurable", None) and getattr(config.configurable, "thread_id", None)
        except Exception:
            thread_id = None

    if thread_id is not None:
        set_thread_id(thread_id)

    # secure user_id
    set_user_id(DEFAULT_USER_ID)

    try:
        # prefer the asynchronous version if it exists
        if hasattr(runnable, "ainvoke") and callable(getattr(runnable, "ainvoke")):
            result = await runnable.ainvoke(state, config=config)
        else:
            # run synchronous invoke in executor to not block
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: runnable.invoke(state, config=config))
        return {"messages": result}
    except Exception as e:
        return {"messages": [{"type": "error", "text": f"sales_assistant error: {e}"}]}



def support_assistant(state: State, config: RunnableConfig) -> dict:
    return {"messages": []}
