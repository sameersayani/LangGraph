Project Overview

Description: Simple examples demonstrating LangGraph workflows (state graph nodes, routers, loops). Use the notebooks to explore routing, conditional edges, and looping graphs.
Notebooks

MultipleInputs.ipynb: Demonstrates choosing operations (sum vs. product) and wiring node behavior. See MultipleInputs.ipynb.
ConditionalAgent.ipynb: Two-stage conditional flow example (router → branch → router2). See ConditionalAgent.ipynb.
Looping.ipynb: Automatic guessing game using a looped router (setup → guess → hint → update_bounds → check_finish → finish). See Looping.ipynb.
Other demos: LoopingCounter.ipynb, LoopingGame.ipynb, Personalized.ipynb, SequentialAgent.ipynb, Hello.ipynb.
Quick Setup (Windows)

Python: 3.8+ recommended.
Create and activate a venv, then install deps:
How to Run

Open the notebook in VS Code or Jupyter Lab.
Run cells in order. Important: run the cell that defines functions/types, then the cell that builds the graph (graph = StateGraph(...) + app = graph.compile()), then any invoke/test cells.
To render the graph, run the cell with display(Image(app.get_graph().draw_mermaid_png())).
Common Gotchas & Tips

TypedDict keys: Add every runtime key you need (e.g., secret, current_guess, result) to the AgentState TypedDict — the framework may enforce the schema and strip unknown keys between nodes.
Router nodes: When using add_conditional_edges, register a router node that returns the state (e.g., graph.add_node("router", lambda state: state)) and pass the selector function only to add_conditional_edges. The selector should return the exact keys used in the mapping.
Node return types: Non-router nodes must return the updated state (a dict). Router/selector functions should return the key string only.
Order matters: Always define functions/types, then compile the graph, then invoke. If you edit functions, re-run the function-definition cell and recompile before invoking.
Debugging: Add short print() statements in nodes to trace execution order. Use small test state dicts to unit-test nodes before wiring to the graph.
Developer Notes

Main files: requirements.txt lists runtime deps.
Flow sanity: For looped graphs, ensure check_finish returns keys that map to either the continuation node or the finish node; ensure finish → END is connected.
Contact / Next Steps