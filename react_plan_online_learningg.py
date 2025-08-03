import inspect
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

import numpy as np # Thêm thư viện để lấy mẫu từ phân phối Beta

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableSequence,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from langgraph.errors import ErrorCode, create_error_message
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.utils.runnable import RunnableCallable, RunnableLike

StructuredResponse = Union[dict, BaseModel]
StructuredResponseSchema = Union[dict, type[BaseModel]]
F = TypeVar("F", bound=Callable[..., Any])


from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,  # For schema validation
)

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# Import necessary LangGraph components
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableSequence,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.utils.runnable import RunnableCallable, RunnableLike
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

StructuredResponse = Union[dict, BaseModel]
StructuredResponseSchema = Union[dict, type[BaseModel]]
F = TypeVar("F", bound=Callable[..., Any])

# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with `add_messages` reducer
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps


class AgentStatePydantic(BaseModel):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: RemainingSteps = 25


class AgentStateWithStructuredResponse(AgentState):
    """The state of the agent with a structured response."""

    structured_response: StructuredResponse


class AgentStateWithStructuredResponsePydantic(AgentStatePydantic):
    """The state of the agent with a structured response."""

    structured_response: StructuredResponse


StateSchema = TypeVar("StateSchema", bound=Union[AgentState, AgentStatePydantic])
StateSchemaType = Type[StateSchema]

PROMPT_RUNNABLE_NAME = "Prompt"

Prompt = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], LanguageModelInput],
    Runnable[StateSchema, LanguageModelInput],
]


def _get_state_value(state: StateSchema, key: str, default: Any = None) -> Any:
    return (
        state.get(key, default)
        if isinstance(state, dict)
        else getattr(state, key, default)
    )


def _get_prompt_runnable(prompt: Optional[Prompt]) -> Runnable:
    prompt_runnable: Runnable
    if prompt is None:
        prompt_runnable = RunnableCallable(
            lambda state: _get_state_value(state, "messages"), name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, str):
        _system_message: BaseMessage = SystemMessage(content=prompt)
        prompt_runnable = RunnableCallable(
            lambda state: [_system_message] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, SystemMessage):
        prompt_runnable = RunnableCallable(
            lambda state: [prompt] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    elif inspect.iscoroutinefunction(prompt):
        prompt_runnable = RunnableCallable(
            None,
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif callable(prompt):
        prompt_runnable = RunnableCallable(
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, Runnable):
        prompt_runnable = prompt
    else:
        raise ValueError(f"Got unexpected type for `prompt`: {type(prompt)}")

    return prompt_runnable


def _should_bind_tools(
    model: LanguageModelLike, tools: Sequence[BaseTool], num_builtin: int = 0
) -> bool:
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if not isinstance(model, RunnableBinding):
        return True

    if "tools" not in model.kwargs:
        return True

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools) - num_builtin:
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_react_agent must match"
            f" Got {len(tools)} tools, expected {len(bound_tools) - num_builtin}"
        )

    tool_names = set(tool.name for tool in tools)
    bound_tool_names = set()
    for bound_tool in bound_tools:
        # OpenAI-style tool
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        # Anthropic-style tool
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            # unknown tool type so we'll ignore it
            continue

        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in the model.bind_tools()")

    return False


def _get_model(model: LanguageModelLike) -> BaseChatModel:
    """Get the underlying model from a RunnableBinding or return the model itself."""
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if isinstance(model, RunnableBinding):
        model = model.bound

    if not isinstance(model, BaseChatModel):
        raise TypeError(
            f"Expected `model` to be a ChatModel or RunnableBinding (e.g. model.bind_tools(...)), got {type(model)}"
        )

    return model


def _validate_chat_history(
    messages: Sequence[BaseMessage],
) -> None:
    """Validate that all tool calls in AIMessages have a corresponding ToolMessage."""
    all_tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    tool_call_ids_with_results = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    tool_calls_without_results = [
        tool_call
        for tool_call in all_tool_calls
        if tool_call["id"] not in tool_call_ids_with_results
    ]
    if not tool_calls_without_results:
        return

    error_message = create_error_message(
        message="Found AIMessages with tool_calls that do not have a corresponding ToolMessage. "
        f"Here are the first few of those tool calls: {tool_calls_without_results[:3]}.\n\n"
        "Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage "
        "(result of a tool invocation to return to the LLM) - this is required by most LLM providers.",
        error_code=ErrorCode.INVALID_CHAT_HISTORY,
    )
    raise ValueError(error_message)


# --- Lớp AdaptivePlanner được cập nhật với Thompson Sampling ---
class AdaptivePlanner:
    def __init__(self, initial_tool_success_rates: dict = None, initial_refine_threshold: float = 0.5):
        # Lưu trữ alpha và beta cho mỗi tool để cập nhật Bayes
        self.tool_beta_params = {}

        if initial_tool_success_rates:
            for tool_name, rate in initial_tool_success_rates.items():
                total_initial_observations = 10
                alpha_init = max(1, round(rate * total_initial_observations))
                beta_init = max(1, round((1 - rate) * total_initial_observations))
                self.tool_beta_params[tool_name] = {"alpha": alpha_init, "beta": beta_init}

        self.refine_threshold = initial_refine_threshold
        print("Adaptive Planner initialized with Thompson Sampling.")

    def _get_tool_success_rate_thompson_sampling(self, tool_name: str) -> float:
        # Sử dụng prior đồng nhất Beta(1,1) nếu tool chưa được biết
        params = self.tool_beta_params.get(tool_name, {"alpha": 1, "beta": 1})
        alpha = params["alpha"]
        beta = params["beta"]
        # Lấy mẫu một giá trị từ phân phối Beta
        return np.random.beta(alpha, beta)

    def _get_mean_success_rate(self, tool_name: str) -> float:
        params = self.tool_beta_params.get(tool_name, {"alpha": 1, "beta": 1})
        alpha = params["alpha"]
        beta = params["beta"]
        return alpha / (alpha + beta)
        
    def _extract_features(self, state: dict) -> Any:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        features = {
            "has_tool_calls_in_last_ai_message": isinstance(last_message, AIMessage) and bool(last_message.tool_calls),
            "tool_calls_count": len(last_message.tool_calls) if isinstance(last_message, AIMessage) and last_message.tool_calls else 0
        }
        return features

    def decide_next_action_type(self, state: dict) -> Literal["call_tool", "finish", "refine_prompt"]:
        features = self._extract_features(state)
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        
        if features["has_tool_calls_in_last_ai_message"]:
            tool_calls = last_message.tool_calls
            
            # Thực hiện Thompson Sampling cho mỗi tool được gợi ý
            sampled_rates = {
                tool_call["name"]: self._get_tool_success_rate_thompson_sampling(tool_call["name"])
                for tool_call in tool_calls
            }
            
            # Chọn tool có sampled rate cao nhất
            best_tool_name = max(sampled_rates, key=sampled_rates.get)
            best_sampled_rate = sampled_rates[best_tool_name]
            
            # So sánh sampled rate tốt nhất với ngưỡng tinh chỉnh
            if best_sampled_rate > self.refine_threshold:
                # Log thông tin để gỡ lỗi
                all_mean_rates = {name: self._get_mean_success_rate(name) for name in sampled_rates.keys()}
                print(f"Adaptive Planner decides to CALL_TOOL: {best_tool_name} "
                      f"(sampled rate: {best_sampled_rate:.2f}). "
                      f"Mean rates: {all_mean_rates}")
                return "call_tool"
            
            print(f"Adaptive Planner decides to REFINE_PROMPT (best sampled rate: {best_sampled_rate:.2f} < threshold).")
            return "refine_prompt"
            
        elif isinstance(last_message, AIMessage) and not features["has_tool_calls_in_last_ai_message"]:
            if "sorry" in last_message.content.lower() or "error" in last_message.content.lower():
                print("Adaptive Planner decides to REFINE_PROMPT (LLM indicates issue).")
                return "refine_prompt"
            
            print("Adaptive Planner decides to FINISH_TASK.")
            return "finish"
        
        print("Adaptive Planner defaults to REFINE_PROMPT (initial or unclear state).")
        return "refine_prompt"

    def provide_feedback(self, tool_name: str, success: bool):
        if tool_name not in self.tool_beta_params:
            self.tool_beta_params[tool_name] = {"alpha": 1, "beta": 1}

        if success:
            self.tool_beta_params[tool_name]["alpha"] += 1
        else:
            self.tool_beta_params[tool_name]["beta"] += 1
        
        current_mean_success_rate = self._get_mean_success_rate(tool_name)

        all_mean_success_rates = [self._get_mean_success_rate(name) for name in self.tool_beta_params.keys()]
        avg_mean_success_rate = sum(all_mean_success_rates) / len(all_mean_success_rates) if all_mean_success_rates else 0.5
        self.refine_threshold = max(0.2, min(0.8, 0.5 + (0.5 - avg_mean_success_rate) * 0.2)) 

        print(f"Adaptive Planner received feedback for {tool_name}: {'Success' if success else 'Failure'}.")
        print(f"Updated mean success rate for {tool_name}: {current_mean_success_rate:.2f} (alpha={self.tool_beta_params[tool_name]['alpha']}, beta={self.tool_beta_params[tool_name]['beta']}).")
        print(f"New refine threshold: {self.refine_threshold:.2f}")


def create_react_agent_with_adaptive_planner( 
    model: Union[str, LanguageModelLike],
    tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
    *,
    adaptive_planner: Optional[AdaptivePlanner] = None,
    prompt: Optional[Prompt] = None,
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[StateSchemaType] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: Optional[str] = None,
) -> CompiledStateGraph:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.
    
    This modified version allows for an optional adaptive planner to influence decision making and learn from tool feedback.
    """
    if version not in ("v1", "v2"):
        raise ValueError(
            f"Invalid version {version}. Supported versions are 'v1' and 'v2'."
        )

    if state_schema is not None:
        required_keys = {"messages", "remaining_steps"}
        if response_format is not None:
            required_keys.add("structured_response")

        schema_keys = set(get_type_hints(state_schema))
        if missing_keys := required_keys - set(schema_keys):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if state_schema is None:
        state_schema = (
            AgentStateWithStructuredResponse
            if response_format is not None
            else AgentState
        )

    llm_builtin_tools: list[dict] = []
    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        llm_builtin_tools = [t for t in tools if isinstance(t, dict)]
        tool_node = ToolNode([t for t in tools if not isinstance(t, dict)])
        tool_classes = list(tool_node.tools_by_name.values())

    if isinstance(model, str):
        try:
            from langchain.chat_models import (  # type: ignore[import-not-found]
                init_chat_model,
            )
        except ImportError:
            raise ImportError(
                "Please install langchain (`pip install langchain`) to use '<provider>:<model>' string syntax for `model` parameter."
            )

        model = cast(BaseChatModel, init_chat_model(model))

    tool_calling_enabled = len(tool_classes) > 0

    if (
        _should_bind_tools(model, tool_classes, num_builtin=len(llm_builtin_tools))
        and len(tool_classes + llm_builtin_tools) > 0
    ):
        model = cast(BaseChatModel, model).bind_tools(tool_classes + llm_builtin_tools)  # type: ignore[operator]

    # Sử dụng _get_prompt_runnable gốc. Hàm này sẽ lấy 'state' làm đầu vào và trả về 'list[BaseMessage]'
    model_runnable = _get_prompt_runnable(prompt) | model

    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        is_last_step = _get_state_value(state, "is_last_step", False)
        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )

    def _get_model_input_state(state: StateSchema) -> StateSchema:
        if pre_model_hook is not None:
            messages = (
                _get_state_value(state, "llm_input_messages")
            ) or _get_state_value(state, "messages")
            error_msg = f"Expected input to call_model to have 'llm_input_messages' or 'messages' key, but got {state}"
        else:
            messages = _get_state_value(state, "messages")
            error_msg = (
                f"Expected input to call_model to have 'messages' key, but got {state}"
            )

        if messages is None:
            raise ValueError(error_msg)

        _validate_chat_history(messages)
        # we're passing messages under `messages` key, as this is expected by the prompt
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            state.messages = messages  # type: ignore
        else:
            state["messages"] = messages  # type: ignore

        return state

    # Define the function that calls the model
    # Truyền toàn bộ 'state' dictionary cho model_runnable.invoke()
    def call_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        state = _get_model_input_state(state) # Cập nhật state nếu có pre_model_hook
        response = cast(AIMessage, model_runnable.invoke(state, config))
        # add agent name to the AIMessage
        response.name = name

        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        state = _get_model_input_state(state) # Cập nhật state nếu có pre_model_hook
        response = cast(AIMessage, await model_runnable.ainvoke(state, config))
        # add agent name to the AIMessage
        response.name = name
        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    input_schema: StateSchemaType
    if pre_model_hook is not None:
        # Dynamically create a schema that inherits from state_schema and adds 'llm_input_messages'
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            # For Pydantic schemas
            from pydantic import create_model

            input_schema = create_model(
                "CallModelInputSchema",
                llm_input_messages=(list[AnyMessage], ...),
                __base__=state_schema,
            )
        else:
            # For TypedDict schemas
            class CallModelInputSchema(state_schema):  # type: ignore
                llm_input_messages: list[AnyMessage]

            input_schema = CallModelInputSchema
    else:
        input_schema = state_schema

    def generate_structured_response(
        state: StateSchema, config: RunnableConfig
    ) -> StateSchema:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = model_with_structured_output.invoke(messages, config)
        return {"structured_response": response}

    async def agenerate_structured_response(
        state: StateSchema, config: RunnableConfig
    ) -> StateSchema:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = await model_with_structured_output.ainvoke(messages, config)
        return {"structured_response": response}

    if not tool_calling_enabled:
        # Define a new graph
        workflow = StateGraph(state_schema, config_schema=config_schema)
        workflow.add_node(
            "agent",
            RunnableCallable(call_model, acall_model),
            input_schema=input_schema,
        )
        if pre_model_hook is not None:
            workflow.add_node("pre_model_hook", pre_model_hook)  # type: ignore[arg-type]
            workflow.add_edge("pre_model_hook", "agent")
            entrypoint = "pre_model_hook"
        else:
            entrypoint = "agent"

        workflow.set_entry_point(entrypoint)

        if post_model_hook is not None:
            workflow.add_node("post_model_hook", post_model_hook)  # type: ignore[arg-type]
            workflow.add_edge("agent", "post_model_hook")

        if response_format is not None:
            workflow.add_node(
                "generate_structured_response",
                RunnableCallable(
                    generate_structured_response,
                    agenerate_structured_response,
                ),
            )
            if post_model_hook is not None:
                workflow.add_edge("post_model_hook", "generate_structured_response")
            else:
                workflow.add_edge("agent", "generate_structured_response")

        return workflow.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )

    # Define the function that determines whether to continue or not
    def should_continue(state: StateSchema) -> Union[str, list[Send]]:
        messages = _get_state_value(state, "messages")
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            if post_model_hook is not None:
                return "post_model_hook"
            elif response_format is not None:
                return "generate_structured_response"
            else:
                return END
        # Otherwise if there is, we continue
        else:
            if version == "v1":
                return "tools"
            elif version == "v2":
                if post_model_hook is not None:
                    return "post_model_hook"
                tool_calls = [
                    tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                    for call in last_message.tool_calls
                ]
                return [Send("tools", [tool_call]) for tool_call in tool_calls]

    # Define a new graph
    workflow = StateGraph(state_schema or AgentState, config_schema=config_schema)

    # Define the two nodes we will cycle between
    workflow.add_node(
        "agent",
        RunnableCallable(call_model, acall_model),
        input_schema=input_schema,
    )
    workflow.add_node("tools", tool_node)

    # Optionally add a pre-model hook node that will be called
    # every time before the "agent" (LLM-calling node)
    if pre_model_hook is not None:
        workflow.add_node("pre_model_hook", pre_model_hook)  # type: ignore[arg-type]
        workflow.add_edge("pre_model_hook", "agent")
        entrypoint = "pre_model_hook"
    else:
        entrypoint = "agent"

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point(entrypoint)

    agent_paths = []
    post_model_hook_paths = [entrypoint, "tools"]

    # Add a post model hook node if post_model_hook is provided
    if post_model_hook is not None:
        workflow.add_node("post_model_hook", post_model_hook)  # type: ignore[arg-type]
        agent_paths.append("post_model_hook")
        workflow.add_edge("agent", "post_model_hook")
    else:
        agent_paths.append("tools")

    # Add a structured output node if response_format is provided
    if response_format is not None:
        workflow.add_node(
            "generate_structured_response",
            RunnableCallable(
                generate_structured_response,
                agenerate_structured_response,
            ),
        )
        if post_model_hook is not None:
            post_model_hook_paths.append("generate_structured_response")
        else:
            agent_paths.append("generate_structured_response")
    else:
        if post_model_hook is not None:
            post_model_hook_paths.append(END)
        else:
            agent_paths.append(END)

    if post_model_hook is not None:

        def post_model_hook_router(state: StateSchema) -> Union[str, list[Send]]:
            """Route to the next node after post_model_hook.

            Routes to one of:
            * "tools": if there are pending tool calls without a corresponding message.
            * "generate_structured_response": if no pending tool calls exist and response_format is specified.
            * END: if no pending tool calls exist and no response_format is specified.
            """

            messages = _get_state_value(state, "messages")
            tool_messages = [
                m.tool_call_id for m in messages if isinstance(m, ToolMessage)
            ]
            last_ai_message = next(
                m for m in reversed(messages) if isinstance(m, AIMessage)
            )
            pending_tool_calls = [
                c for c in last_ai_message.tool_calls if c["id"] not in tool_messages
            ]

            if pending_tool_calls:
                pending_tool_calls = [
                    tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                    for call in pending_tool_calls
                ]
                return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]
            elif isinstance(messages[-1], ToolMessage):
                return entrypoint
            elif response_format is not None:
                return "generate_structured_response"
            else:
                return END

        workflow.add_conditional_edges(
            "post_model_hook",
            post_model_hook_router,  # type: ignore[arg-type]
            path_map=post_model_hook_paths,
        )

    workflow.add_conditional_edges(
        "agent",
        should_continue,  # type: ignore[arg-type]
        path_map=agent_paths,
    )

    # Callback để AdaptivePlanner nhận feedback sau khi tool chạy
    def tool_feedback_callback(state: StateSchema) -> StateSchema:
        if not adaptive_planner:
            return state

        messages = _get_state_value(state, "messages")
        new_tool_messages = [m for m in messages if isinstance(m, ToolMessage) and m.additional_kwargs.get("processed_for_feedback", False) is False]

        for tool_msg in new_tool_messages:
            tool_name = tool_msg.name
            success = "error" not in tool_msg.content.lower() and "fail" not in tool_msg.content.lower() and "not available" not in tool_msg.content.lower()

            adaptive_planner.provide_feedback(tool_name, success)
            tool_msg.additional_kwargs["processed_for_feedback"] = True 
        return state

    workflow.add_node("tool_feedback_node", tool_feedback_callback)


    def route_tool_responses(state: StateSchema) -> str:
        for m in reversed(_get_state_value(state, "messages")):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
                return END
        if isinstance(m, AIMessage) and m.tool_calls:
            if any(call["name"] in should_return_direct for call in m.tool_calls):
                return END
        
        return "tool_feedback_node"

    if should_return_direct:
        workflow.add_conditional_edges(
            "tools", route_tool_responses, path_map=[entrypoint, END, "tool_feedback_node"]
        )
        workflow.add_edge("tool_feedback_node", entrypoint)
    else:
        workflow.add_edge("tools", "tool_feedback_node")
        workflow.add_edge("tool_feedback_node", entrypoint)


    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )


# # Keep for backwards compatibility
# create_tool_calling_executor = create_react_agent

# __all__ = [
#     "create_react_agent",
#     "create_tool_calling_executor",
#     "AgentState",
#     "AgentStatePydantic",
#     "AgentStateWithStructuredResponse",
#     "AgentStateWithStructuredResponsePydantic",
# ]

# --- Ví dụ sử dụng ---
if __name__ == "__main__":

    import os

    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool

    # from langchain_openai import ChatOpenAI

    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    llm = init_chat_model()

    @tool
    def get_current_weather(location: str) -> str:
        """Get the current weather in a given location"""
        if "hanoi" in location.lower():
            return "It's sunny with 30 degrees Celsius in Hanoi."
        elif "london" in location.lower():
            return "It's cloudy with 15 degrees Celsius in London."
        else:
            return "Weather data not available for this location."

    @tool
    def search_web(query: str) -> str:
        """Search the web for a given query."""
        if "capital of france" in query.lower():
            return "The capital of France is Paris."
        # Giả định tool này có thể thất bại đôi khi để minh họa học
        if "unknown" in query.lower():
            return "Error: Could not perform web search for 'unknown'."
        return f"Found some results for '{query}' on the web."

    # Khởi tạo Adaptive Planner
    my_adaptive_planner = AdaptivePlanner(
        initial_tool_success_rates={
            "get_current_weather": 0.8, # Giả định ban đầu
            "search_web": 0.6 # Giả định ban đầu
        },
        initial_refine_threshold=0.6 # Ban đầu tin tưởng vào LLM để refine
    )

    # Khởi tạo agent với Adaptive Planner
    graph = create_react_agent_with_adaptive_planner(
        # ChatOpenAI(model="gpt-4o-mini", temperature=0),
        llm,
        tools=[get_current_weather, search_web],
        prompt="You are a helpful AI assistant. Use tools to answer questions. If you need to search for information, use the 'search_web' tool. If you need weather, use 'get_current_weather'.",
        adaptive_planner=my_adaptive_planner, # Truyền planner vào đây
        debug=True,
        version="v2"
    )

    print("--- Agent with Adaptive Planner (Initial State) ---")
    inputs = {"messages": [HumanMessage(content="What is the weather in London and what is the capital of France?")]}
    print("Running first query (London weather & France capital)...")
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)
    
    # Kiểm tra trạng thái của planner sau lần chạy đầu tiên
    print("\nPlanner's state after first query:")
    # print(my_adaptive_planner.tool_success_rates)
    print(my_adaptive_planner.tool_beta_params)
    print(f"Refine Threshold: {my_adaptive_planner.refine_threshold:.2f}")

    print("\n--- Agent with Adaptive Planner (Simulating failure and adaptation) ---")
    # Chúng ta sẽ đưa ra một query mà 'search_web' sẽ thất bại
    inputs2 = {"messages": [HumanMessage(content="Search for unknown topic and then tell me about weather in Hanoi.")]}
    print("Running second query (search for unknown & Hanoi weather)...")
    for chunk in graph.stream(inputs2, stream_mode="updates"):
        print(chunk)

    # Kiểm tra trạng thái của planner sau lần chạy thứ hai
    print("\nPlanner's state after second query:")
    # print(my_adaptive_planner.tool_success_rates)
    print(my_adaptive_planner.tool_beta_params)
    print(f"Refine Threshold: {my_adaptive_planner.refine_threshold:.2f}")

    print("\n--- Agent with Adaptive Planner (Observe potential adaptation) ---")
    # Chạy lại một câu hỏi có thể dùng search_web để xem planner có điều chỉnh không
    inputs3 = {"messages": [HumanMessage(content="What is the capital of Germany?")]}
    print("Running third query (capital of Germany)...")
    for chunk in graph.stream(inputs3, stream_mode="updates"):
        print(chunk)
    
    print("\nPlanner's state after third query:")
    # print(my_adaptive_planner.tool_success_rates)
    print(my_adaptive_planner.tool_beta_params)
    print(f"Refine Threshold: {my_adaptive_planner.refine_threshold:.2f}")