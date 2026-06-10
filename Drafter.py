from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# This is the global variable to store document contesnt
document_content = ""
# We should use injectors in LangGraph to modify the state of the graph, but for simplicity, we will use a global variable here.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    '''Update the document with the provided content.'''
    global document_content
    document_content = content
    return f"Document updated successfully. Current content: {document_content}"

@tool
def save(filename: str) -> str:
    '''Save the current document to a text file and finish the process.
    
    Args:
        filename: Name of the text file.
    '''
    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(document_content)
        print(f"\n Document saved successfully as {filename}.")
        return f"Document saved successfully as {filename}."
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update, save]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents

    - If the user wants to update or modify content, user the  'update' tool with the complete update content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modification.
                                  
    The current document content is:{document_content}
""")

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"] + [user_message])
    response = model.invoke(all_messages)

    print("\n AI:", getattr(response, "content", response))
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(" USING TOOLS:", [tc["name"] for tc in response.tool_calls])

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    '''Determine if we should continue or end the conversation'''

    messages = state["messages"]

    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # and checks if this is a ToolMessage resulting from save
        if(isinstance(message, ToolMessage) and
           "saved" in message.content.lower() and
           "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
        return "continue"
     
def print_message(messages):
    '''Function I made to print the message in a more readable format'''
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {getattr(message, 'content', message)}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    }
)

app = graph.compile()

def run_document_agent():
    print("\n #### DRAFTER STARTED ####")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])

    print("\n #### DRAFTER FINISH?ED ####")

if __name__ == "__main__":
    run_document_agent()

#You can also use Voice tools: Speech to text Conversion
