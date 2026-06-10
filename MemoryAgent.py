import os
from typing import TypedDict, Dict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process_message(state: AgentState) -> AgentState:
    '''This node will take the request you input and respond with a message'''
    response = llm.invoke(state["message"])
    state["message"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process_message", process_message)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)  
agent = graph.compile()

conversation_history = []
user_input = input("You: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"message": conversation_history})
    print(result['message'])
    conversation_history = result['message']
    user_input = input("You: ")

with open("logging.txt", "w", encoding="utf-8") as f:
    f.write("This is the log of the conversation:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("End of conversation.\n")

with open("logging.txt", "r", encoding="utf-8") as f:
    print(f.read())

print("Conversation logged to logging.txt")