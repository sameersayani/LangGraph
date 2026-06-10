from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain import tools
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    '''This is an addition tool that adds two numbers together'''
    return a + b    

def multiply(a: int, b: int) -> int:
    '''This is a multiplication tool that multiplies two numbers together'''
    return a * b

def subtract(a: int, b: int) -> int:
    '''This is a subtraction tool that subtracts two numbers'''
    return a - b

def divide(a: int, b: int) -> int:
    '''This is a division tool that divides two numbers'''
    if b == 0:
        return "Error: Division by zero is undefined."
    return a / b

tools = [add, multiply, subtract, divide]
llm = ChatOpenAI(model="gpt-4o")
# Create an agent graph (handles tool-calling loop)
system_prompt = SystemMessage(content="You are a helpful assistant that can use tools to answer questions.")
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

def model_call(state: AgentState) -> AgentState:
    # Invoke the compiled agent graph with the current messages.
    inputs = {"messages": state["messages"]}
    result = agent.invoke(inputs)
    return {"messages": result.get("messages", [])}

def should_continue(state: AgentState):
    message = state["messages"]
    last_message = message[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }   
)

graph.add_edge("tools", "our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 100+20, Add 3+4, Multiply 5*6, Subtract 10-4, Divide 20/5, Add 40+12 and then multiply by 6 and also crack a joke" )]}
print_stream(app.stream(inputs, stream_mode="values"))