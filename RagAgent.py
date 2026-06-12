from typing import Annotated, Sequence, TypedDict
import os
import sys
import glob
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# temperation zero to minimize hallucination, it makes output more deterministic
llm = ChatOpenAI(model="gpt-4o", temperature=0) 

# Embedding Model, compatible with the LLM in use
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

# PDF path resolution: prefer CLI arg, then environment variable PDF_PATH, then look for any PDF in the workspace
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    pdf_path = os.environ.get("PDF_PATH")

if not pdf_path:
    # search for any .pdf files next to this script or in the current working directory
    base_dir = os.path.dirname(__file__) or os.getcwd()
    candidates = glob.glob(os.path.join(base_dir, "*.pdf")) + glob.glob(os.path.join(os.getcwd(), "*.pdf"))
    candidates = list(dict.fromkeys(candidates))
    if candidates:
        pdf_path = candidates[0]
        print(f"Auto-discovered PDF: {pdf_path}")

if not pdf_path or not os.path.exists(pdf_path):
    print("PDF file not found. Provide a path as the first argument or set the PDF_PATH environment variable.")
    print("Examples:")
    print("  python RagAgent.py path/to/file.pdf")
    print("  set PDF_PATH=path\\to\\file.pdf && python RagAgent.py")
    sys.exit(1)

pdf_loader = PyPDFLoader(pdf_path)

# Check if PDF present
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loadiing PDF: {e}")
    raise 

# Chunk Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_split = text_splitter.split_documents(pages)
persistent_directory = r"D:\My Projects\LangGraph\Agents"
collection_name = "stock_market"

if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory)

try:
    # Create Chroma database using our embedding model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persistent_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Retriever
retriever= vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs = {"k": 5} # K is amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    '''
    This tool searches and returns the information from the Stock Market Performance document 2026
    '''
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the stock market report"
    
    result = []
    for i, doc in enumerate(docs):
        result.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(result)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    '''Check if the last message contains tool calls'''
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = system_prompt = """
You are an intelligent AI assistant who answers questions about Indian Stock Market Performance in mid year 2026 based on the provided documents.
Use the retriever tool available to answer questions about the stock market performance data. You can use it as needed.
If you need to look up some information before answering or asking a follow-up question, you are allowed to do that.
Please always cite the specific parts of the documents you use in your answers.
"""

# Dictionary of the tools by name
tools_dict = {}
for t in tools:
    name = getattr(t, 'name', None) or getattr(t, '__name__', None) or t.__class__.__name__
    tools_dict[name] = t

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    '''Function to call the LLM with the current state.'''
    # Build the prompt messages: system + conversation so far
    messages = list(state['messages'])
    prompt_messages = [SystemMessage(content=system_prompt)] + messages

    # Invoke the LLM. This should return an assistant message (possibly with tool_calls).
    assistant_response = llm.invoke(prompt_messages)

    # Append the assistant response to the conversation messages and return
    return {'messages': messages + [assistant_response]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    '''Execute tool calls from the LLM's response and append ToolMessage(s) to the conversation.'''
    messages = list(state['messages'])
    assistant_msg = messages[-1]
    tool_calls = getattr(assistant_msg, 'tool_calls', []) or []

    tool_messages = []
    for t in tool_calls:
        print(f"Calling tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if t['name'] not in tools_dict:
            print(f"\n Tool: {t['name']} does not exist")
            result = "Incorrect tool name, please retry and select tool from list of available tools"
        else:
            # invoke the tool and coerce to string
            tool_obj = tools_dict[t['name']]
            try:
                result = tool_obj.invoke(t['args'].get('query', ''))
            except Exception as e:
                result = f"Tool invocation error: {e}"
            print(f"Result length: {len(str(result))}")

        tool_messages.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    # Append tool result messages to the conversation so the model receives them next
    messages.extend(tool_messages)
    print("Tools Execution Complete. Back to the model..")
    return {'messages': messages}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

agent = graph.compile()


def running_agent():
    print("\n **** RAG AGENT ****")
    while True:
        user_input = input("\nAsk me your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        #Convert back to HumanMessage type
        messages = [HumanMessage(content=user_input)]
        result = agent.invoke({'messages': messages})

        print("\n ***** ANSWER ****")
        print(result['messages'][-1].content)

running_agent()