import os
from typing import Annotated, List, Literal, TypedDict

# Các import từ LangChain community
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

# Sử dụng lại các import gốc của bạn
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

# Các import cần thiết cho giải pháp
from pydantic.v1 import BaseModel

langfuse_handler = CallbackHandler()


MAX_RETRIES = 4

# from sqlalchemy import create_engine
# from sqlalchemy.pool import StaticPool
# def get_engine_for_chinook_db():
#     # url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
#     # response = requests.get(url)
#     # sql_script = response.text
#     with open("Chinook_Sqlite.sql", 'r') as file:
#         sql_script = file.read()
#     connection = sqlite3.connect(":memory:", check_same_thread=False)
#     connection.executescript(sql_script)
#     return create_engine("sqlite://", creator=lambda: connection, poolclass=StaticPool)

# # --- 1. SETUP ---

# engine = get_engine_for_chinook_db()
# db = SQLDatabase(engine)

db = SQLDatabase.from_uri(db_url)

llm = init_chat_model()


# --- 2. TẠO CÔNG CỤ THẬT VÀ CÔNG CỤ PROXY ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
all_sql_tools = toolkit.get_tools()

# Lấy ra các công cụ thật để sử dụng sau này và để sao chép thông tin
real_run_query_tool = next(t for t in all_sql_tools if t.name == "sql_db_query")
list_tables_tool = next(t for t in all_sql_tools if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in all_sql_tools if t.name == "sql_db_schema")
checker = next(t for t in all_sql_tools if t.name == "sql_db_query_checker")



# Tạo một công cụ Proxy an toàn với return_direct=True
# Nó sẽ "giả mạo" công cụ chạy query thật
class ProxyQueryTool(BaseTool):
    name: str = "sql_db_query" # Giữ nguyên tên để agent không bị bối rối
    description: str = real_run_query_tool.description # SAO CHÉP description thật
    args_schema: type[BaseModel] = real_run_query_tool.args_schema # SAO CHÉP schema thật
    return_direct: bool = True # ĐÂY LÀ CHÌA KHÓA ĐỂ DỪNG AGENT

    def _run(self, query: str) -> str:
        """Hàm này chỉ trả về câu query, KHÔNG thực thi nó."""
        return query

# Khởi tạo công cụ proxy
proxy_query_tool = ProxyQueryTool()

# Đây là danh sách công cụ cuối cùng sẽ được đưa cho agent
tools_for_agent = [retriever_tool,list_tables_tool, get_schema_tool,checker, proxy_query_tool]

# --- 3. STATE VÀ AGENT ---
class AgentState(TypedDict, total=False):
    # Annotated giúp LangGraph biết cách kết hợp tin nhắn (thêm vào danh sách)
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    query_to_run: str # Key mới để lưu câu query cần kiểm tra
    retry_count: int

# Prompt có thể giữ đơn giản vì agent sẽ dùng công cụ theo `description` You are a SQL expert helping sales staff find products and assisting sales staff
top_k=5
generate_query_system_prompt = f"""
You are an expert SQL agent. Your goal is to answer the user's question by interacting with a database.

**PLANNING:**
1. Always start by using `sql_db_list_tables` to understand the database.
2. Then, use `sql_db_schema` on the most relevant tables to understand their structure.
3. Based on your plan, formulate a syntactically correct {db.dialect} query.
4. If a query does not provide enough information, continue planning and execute more queries until you can answer the question.

**QUERY RULES:**
- NEVER query for all columns. Only select the relevant columns.
- Unless the user asks for everything, LIMIT your query to at most {top_k} results.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.).

"""

suffix = (

)
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generate_query_system_prompt),
        ("placeholder", "{messages}"),
    ]
)


# Agent này thông minh, nhưng công cụ sql_db_query của nó đã bị tráo đổi
config = {"recursion_limit": 50}
react_agent_runnable = create_react_agent(
    llm,
    tools=tools_for_agent,
    prompt=agent_prompt,
)

# --- 4. ĐỊNH NGHĨA CÁC NODE MỚI ---
# --- NODES ĐÃ SỬA LỖI TRIỆT ĐỂ ---

def planning_agent_node(state: AgentState):
    """Node này cách ly `react_agent_runnable` để bảo toàn state.
    Đây là cách làm đúng và an toàn nhất.
    """
    print("\n---NODE: Planning Agent---")
    
    # 1. BẢO VỆ STATE: Lưu lại retry_count trước khi gọi "hố đen"
    current_retry = state.get('retry_count', 0)
    
    # 2. CÁCH LY: Chỉ đưa cho agent những gì nó hiểu (messages)
    agent_inputs = {"messages": state["messages"]}
    
    # Ghi lại số lượng tin nhắn để chỉ lấy phần mới
    messages_before = len(agent_inputs["messages"])
    
    # Gọi hộp đen với input đã được cách ly
    result_state_from_agent = react_agent_runnable.invoke(agent_inputs,config=config)
    
    # Chỉ lấy các tin nhắn MỚI được tạo ra
    new_messages = result_state_from_agent["messages"][messages_before:]
    
    # Tạo dictionary cập nhật trả về
    update_dict = {
        "messages": new_messages,
        "retry_count": current_retry # 3. KHÔI PHỤC: Đưa retry_count đã lưu trở lại
    }
    
    last_new_message = new_messages[-1] if new_messages else None
    if isinstance(last_new_message, ToolMessage) and last_new_message.name == "sql_db_query":
        query = last_new_message.content
        print(f">>> Agent đã gửi yêu cầu chạy query qua Proxy: '{query}'")
        update_dict["query_to_run"] = query
        update_dict["messages"]=new_messages[0:-1]###################
    else:
        print(">>> Agent đã đưa ra câu trả lời cuối cùng.")
        
    return update_dict


def check_and_run_query_node(state: AgentState):
    """Node này kiểm tra và chạy công cụ query THẬT."""
    print("\n---NODE: Check and Run Real Query---")
    query = state.get("query_to_run")
    query=query.replace('"',"'").lower().replace(" like ", " ilike ")
    current_retry = state.get('retry_count', 0) # Lấy retry_count để bảo toàn
    if not query:
        return {"messages": [ToolMessage(content="Error: No query was provided to run.", name="sql_db_query")]}

    print(f"Thực thi câu query đã được kiểm duyệt: '{query}'")
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    try:
        # Sử dụng công cụ thật ở đây
        query_result = real_run_query_tool.invoke({"query": query})
        if query_result in ("[]", "()", ""):
            tool_message = ToolMessage(content="Query executed successfully but returned no results.", name="sql_db_query", tool_call_id=tool_call_id)
        else:
            tool_message = ToolMessage(content=query_result, name="sql_db_query", tool_call_id=tool_call_id)
    except Exception as e:
        print(f"!!! LỖI khi thực thi query: {e}")
        tool_message = ToolMessage(content=f"Execution Error: {e}", name="sql_db_query", tool_call_id=tool_call_id)

    # Xóa query sau khi chạy và trả về kết quả cho agent
    return {"messages": [tool_message], "query_to_run": None, "retry_count": current_retry}


def final_failure_node(state: AgentState):
    return {"messages": [AIMessage(content=f"Xin lỗi, đã thất bại sau {MAX_RETRIES} lần thử.")]}

# --- ROUTERS (Đã đúng, giờ sẽ hoạt động đúng) ---
def route_after_planning(state: AgentState) -> Literal["run_query", "__end__"]:
    print("\n---ROUTER: After Planning---")
    if state.get("query_to_run"): return "run_query"
    else: return "__end__"

def route_after_query(state: AgentState) -> Literal["planning_agent", "final_failure"]:
    print("\n---ROUTER: After Query Execution---")
    last_message_content = state["messages"][-1].content
    is_failure = "error" in last_message_content.lower() or "no results" in last_message_content.lower()
    
    # Router chỉ đọc state và đưa ra quyết định, không thay đổi nó
    current_retry = state.get('retry_count', 0)
    
    if is_failure:
        # Tăng giá trị để kiểm tra điều kiện
        next_retry_attempt = current_retry + 1
        print(f"Thất bại được ghi nhận. Thử lại lần {next_retry_attempt}/{MAX_RETRIES}.")
        if next_retry_attempt >= MAX_RETRIES:
            print(">>> Quyết định: Đạt giới hạn thử lại.")
            return "final_failure"
        else:
            print(">>> Quyết định: Thử lại.")
            return "planning_agent"
    else:
        print(">>> Quyết định: Query thành công.")
        return "planning_agent"

# --- NODE CẬP NHẬT STATE RIÊNG BIỆT (GIẢI PHÁP AN TOÀN NHẤT) ---

def update_retry_node(state: AgentState) -> dict:
    """Node này có một nhiệm vụ duy nhất: cập nhật `retry_count` một cách an toàn.
    Nó được gọi sau khi router quyết định cần phải thử lại.
    """
    print("\n---NODE: Update Retry Count---")
    last_message_content = state["messages"][-1].content
    is_failure = "error" in last_message_content.lower() or "no results" in last_message_content.lower()
    
    if is_failure:
        current_retry = state.get('retry_count', 0) + 1
        print(f"Tăng bộ đếm lỗi lên: {current_retry}")
        return {"retry_count": current_retry}
    else:
        # Nếu thành công, reset bộ đếm
        print("Query thành công cũng tăng retry lên 1.")
        # return {"retry_count": 0}
        current_retry = state.get('retry_count', 0) + 1
        return {"retry_count": current_retry}

# --- GRAPH & CHẠY THỬ ---
class InitialRetrievalNode:
    """Node để thực thi bước retrieval bắt buộc ban đầu."""
    def __init__(self, retriever):
        self.retriever = retriever

    def __call__(self, state: AgentState):
        """Trả về kết quả retrieval dưới dạng ToolMessage để agent xử lý."""
        print("\n--- STAGE 1: FORCED RETRIEVAL ---")
        print(state["messages"])
        user_question = state["messages"][-1]["content"][-1]["text"]
        retrieved_docs = self.retriever.invoke(user_question)
        # context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        # if not context:
        #     context = "No additional information found in the knowledge base."

        # # Tạo một ToolMessage để đưa kết quả retrieval vào cuộc hội thoại.
        # # Chúng ta cần một tool_call_id giả để nó hợp lệ.
        retrieval_message = ToolMessage(
            content=retrieved_docs,
            tool_call_id="initial_retrieval_call" # ID giả, chỉ cần tồn tại
        )
        # Để agent hiểu, chúng ta cần nói cho nó biết "đây là kết quả của tool nào"
        # bằng cách thêm một AIMessage giả yêu cầu tool này.
        fake_ai_call = AIMessage(
            content="Tôi nhận được câu hỏi từ user, Tôi sẽ sử dụng công cụ initial_retrieval để tìm kiếm các từ chính xác, sau đó gọi tool sql_db_list_tables và viết query ",
            tool_calls=[{"name": "initial_retrieval", "args": {}, "id": "initial_retrieval_call"}]
        )
        return {"messages": [fake_ai_call, retrieval_message]}

retrieval_node_runner = InitialRetrievalNode(retriever=retriever_tool)


builder = StateGraph(AgentState)
builder.add_node("planning_agent", planning_agent_node)
builder.add_node("run_query", check_and_run_query_node)
# Thêm node cập nhật mới vào graph
builder.add_node("update_retry_count", update_retry_node)
builder.add_node("final_failure", final_failure_node)

builder.add_node("initial_retrieval", retrieval_node_runner)

# builder.set_entry_point("planning_agent")
builder.set_entry_point("initial_retrieval")
builder.add_edge("initial_retrieval", "planning_agent")
builder.add_conditional_edges("planning_agent", route_after_planning)

# Thay đổi luồng: sau khi chạy query, đi đến router
builder.add_conditional_edges(
    "run_query",
    route_after_query,
    {
        "planning_agent": "update_retry_count", # Nếu thành công, đi cập nhật (reset)
        "final_failure": "final_failure",
    }
)

# Sau khi cập nhật retry, LUÔN LUÔN đi đến agent để lên kế hoạch lại
builder.add_edge("update_retry_count", "planning_agent")
builder.add_edge("final_failure", END)

master_agent = builder.compile().with_config(config={"callbacks": [langfuse_handler]})
# # --- 7. CHẠY THỬ ---


# # print("\n\n--- TEST CASE 1: KẾ HOẠCH NHIỀU BƯỚC (THÀNH CÔNG) ---")
# # inputs1 = AgentState(messages=[HumanMessage(content="How many albums does Alice In Chains have and Which country's customers spent the most?")], retry_count=0)
# # for s in master_agent.stream(inputs1, config=config,stream_mode="values",):
# #     # print(s)
# #     s["messages"][-1].pretty_print()
# #     print("-----\n")

# # print("\n\n--- TEST CASE 2: LỖI CỐ Ý & THỬ LẠI ---")
# # inputs2 = AgentState(messages=[HumanMessage(content="How many employees are there?")], retry_count=0)
# # for s in master_agent.stream(inputs2, config=config,stream_mode="values",):
# #     # print(s)
# #     s["messages"][-1].pretty_print()
# #     print("-----\n")


# def stream_graph_updates(user_input: str):
#     # initial_state = {
#     #         "original_question": user_input,
#     #         "retrieved_context": "", # Khởi tạo rỗng
#     #         "final_answer": "",      # Khởi tạo rỗng
#     #         "db": db,
#     #         "llm": llm,
#     #         "retriever": retriever_tool
#     #     }

#     for event in master_agent.stream(AgentState(messages=[HumanMessage(content=user_input)], retry_count=0),stream_mode="values",):
#         # for value in event.values():
#         #     print("Assistant:", value["messages"][-1].content)
#         event["messages"][-1].pretty_print()
#         # print(event.values())

# while True:
#     # try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break

#         stream_graph_updates(user_input)
# # First, list all tables. Then, tell me how many tracks are in the 'Rock' genre.
# # What is the average length from the non_existent_table?

# from IPython.display import Image, display

# try:
#     display(Image(agent_executor.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass