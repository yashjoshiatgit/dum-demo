from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage 
from prompts.prompt import ORCHESTRATOR_PROMPT
from langchain_core.messages import AIMessage  
from config import llm
from tools.slack_agent import Slack_Agent
from tools.ticket_agent import Jira_Agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
import json

Orchestrator_Agent = create_agent(llm, [], system_prompt=ORCHESTRATOR_PROMPT)

def _append_steps(existing: List[str], update: List[str]) -> List[str]:
    return existing + update

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: Optional[str]
    ticket_id: Optional[str]
    completed_step: Annotated[List[str], _append_steps]  # ← Safe concatenation
    task: Optional[str]

def slack_agent_node(state: GraphState):
    print("&&&&&&&&&&&&&&&&&&==SLACK_AGENT==&&&&&&&&&&&&&&&&&&")
    task = state.get("task", "Handle Slack communication Default")
    print(f"Task:{task}")
    response = Slack_Agent.invoke({"messages": task}) 
    print(f"Response Fron Slack Agent : {response}")
    new_step = f"SlackAgent_Executed: {task}"
    print(f"new_step:{new_step}")
    return {"completed_step": [new_step]}


def ticket_agent_node(state: GraphState):
    print("&&&&&&&&&&&&&&&&&&==JIRA_AGENT==&&&&&&&&&&&&&&&&&&")
    task = state.get("task", "Handle Jira ticket operations Default")
    print(f"Task:{task}")
    response = Jira_Agent.invoke({"messages": task})
    print(f"Response Fron Ticket Agent : {response}")
    new_step = f"TicketAgent_Executed: {task}"
    print(f"new_step:{new_step}")
    return {"completed_step": [new_step]}


def orchestrator_node(state: GraphState) -> dict:
    print("&&&&&&&&&&&&&&&&&&==ORCHESTRATOR_AGENT==&&&&&&&&&&&&&&&&&&")
    history = state.get("messages", [])
    print(f"History: {history}")
    completed_steps = state.get('completed_step', [])
    print(f"comleted_tasks: {completed_steps}")

    flat_steps = []
    for step in completed_steps:
        if isinstance(step, str):
            flat_steps.append(step)
        elif isinstance(step, list):
            flat_steps.extend([s for s in step if isinstance(s, str)])

    completed_steps_str = " | ".join(flat_steps) if flat_steps else "None"
    print(f"Completed Task's String : {completed_steps_str}")
    task_for_agent = ""
    decision = "END"

    try:
        print("--------------------------------------KEY AREA----------------------------------------")
        print(f"History : {history}")

        # First, generate the next small task based on history and completed steps
        task_instruction = (
            f"Completed steps: {completed_steps_str}\n"
            "Based on the conversation flow, generate the next precise, small, goal-oriented task. "
            "Only return the task instruction, no explanations. "
            "It should be small and focused on one actionable step. No extra words."
        )
        task_for_agent = llm.invoke(history + [HumanMessage(content=task_instruction)]).content.strip()
        print(f"Generated task: {task_for_agent}")

        if task_for_agent:
            # Now, decide which agent to assign the task to, based on the task and history
            agent_instruction = (
                f"Next task: {task_for_agent}\n"
                "Decide which agent should handle this task: SlackAgent, TicketAgent, or END if no more tasks."
            )
            response = Orchestrator_Agent.invoke({"messages": history + [HumanMessage(content=agent_instruction)]})
            print(f"response: {response}")
            decision = response["messages"][-1].content.strip()
            print(f"decision : {decision}")

            if decision not in ["SlackAgent", "TicketAgent", "END"]:
                logging.warning(f"Invalid decision: {decision}. Defaulting to END.")
                decision = "END"
        else:
            decision = "END"

        print("--------------------------------------KEY AREA----------------------------------------")
    except Exception as e:
        logging.exception("Orchestrator failed.")
        decision = "END"

    print(f"Decision: {decision}")
    print(f"Task: {task_for_agent}")

    return {
        "task": task_for_agent,
        "next": decision,
        "messages": [AIMessage(content=f"Decided: {decision} | Task: {task_for_agent}")]
    }

graph = StateGraph(GraphState)

graph.add_node("Orchestrator", orchestrator_node)
graph.add_node("SlackAgent", slack_agent_node)
graph.add_node("TicketAgent", ticket_agent_node)

graph.add_conditional_edges(
    "Orchestrator",
    lambda s: s["next"], 
    {
        "SlackAgent": "SlackAgent",
        "TicketAgent": "TicketAgent",
        "END": END,
    },
)

graph.add_edge("SlackAgent", "Orchestrator")
graph.add_edge("TicketAgent", "Orchestrator")

graph.set_entry_point("Orchestrator")
workflow = graph.compile(checkpointer=MemorySaver())
workflow.get_graph().print_ascii()

# def slack_agent_node(state: GraphState):
#     #Here the Task can be e.x -Send slack message with this information using tools and informations
#     task = state.get("task", "Handle Slack communication Defalut ")
#     # logging.info(f"SlackAgent: Processing state for task '{task}'")
#     response = Slack_Agent.invoke({"messages": task})
#     # print(f"task: {state.get('task', [])}")
#     # print(f"next: {state.get('messages', [])}")
#     # print(f"messages: {state.get('next', [])}")
#     state['completed_step'].append("task")
#     return {"comleted_step": state["completed_step"]}

# def ticket_agent_node(state: GraphState):
#     #Here the Task can be e.x -Send jira message or approva ticcket with this information using tools and informations
#     task = state.get("task", "Handle Jira ticket operations Defalut")
#     # logging.info(f"TicketAgent: Processing state for task '{task}'")
#     response = Jira_Agent.invoke({"messages": task})
#     # print(f"task: {state.get('task', [])}")
#     # print(f"next: {state.get('messages', [])}")
#     # print(f"messages: {state.get('next', [])}")
#     state['completed_step'].append("task")
#     return {"comleted_step": state["completed_step"]}
    


# def orchestrator_node(state: GraphState) -> dict:
#     """
#     Determines the next agent to run based on the conversation history.
#     """
#     history = state.get("messages", []) or []
#     task_pending = state.get('completed_step', []) or []
    
#     try:
#         response = Orchestrator_Agent.invoke({"messages": history})
#         decision = response["messages"][-1].content.strip()

#         if decision not in ["SlackAgent", "TicketAgent", "END"]:
#             logging.warning(f"Orchestrator returned an unrecognized decision: '{decision}'. Defaulting to END.")
#             decision = "END"
            
#     except Exception:
#         logging.exception("Orchestrator agent invocation failed. Defaulting to END.")
#         decision = "END"

#     task_for_agent = llm.invoke(task_pending + "here are the completed task now give the next task instruction based on main task" + state["messages"]).content

#     print(f"::::::::::::::::::::::::::::::::::::::::{decision};;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
#     print(f"::::::::::::::::::::::::::::::::::::::::{task_for_agent};;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
#     return {
#         "task": task_for_agent,
#         "next": decision
#     }

# graph = StateGraph(GraphState)

# graph.add_node("Orchestrator", orchestrator_node)
# graph.add_node("SlackAgent", slack_agent_node)
# graph.add_node("TicketAgent", ticket_agent_node)

# graph.add_conditional_edges(
#     "Orchestrator",
#     lambda s: s["next"],
#     {
#         "SlackAgent": "SlackAgent",
#         "TicketAgent": "TicketAgent",
#         "END": END,
#     },
# )

# graph.add_edge("SlackAgent", "Orchestrator")
# graph.add_edge("TicketAgent", "Orchestrator")

# graph.set_entry_point("Orchestrator")
# workflow = graph.compile(checkpointer=MemorySaver())
# workflow.get_graph().print_ascii()
