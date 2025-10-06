from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import Literal
import json
import os
from dotenv import load_dotenv
import uuid

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ---------------------- DATA MODELS ----------------------
class Route(BaseModel):
    event_type: Literal["corporate", "private", "wedding"] = Field(
        None, description="The type of event for routing the catering request"
    )

class CateringState(TypedDict):
    event_date: str
    headcount: int
    menu: List[str]
    event_type: Optional[str]
    normalized_request: Optional[Dict]
    quote: Optional[Dict]
    approval_status: Optional[str]
    reason: Optional[str]

# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

# ---------------------- NODES ----------------------
def capture_request(state: CateringState):
    """Normalize and summarize user inputs."""
    summary = {
        "date": state["event_date"],
        "headcount": state["headcount"],
        "menu_items": state["menu"],
    }
    print(" Captured request:", summary)
    return {"normalized_request": summary}

def route_event_type(state: CateringState):
    """Route the input to the appropriate event type node."""
    decision = router.invoke(
        [
            SystemMessage(
                content="Determine the event type (corporate, private, or wedding) based on the request. "
                        "Consider headcount (>50 is corporate, <=50 is private) and menu context (e.g., 'wedding cake' suggests wedding)."
            ),
            HumanMessage(content=json.dumps(state["normalized_request"])),
        ]
    )
    print(" Routing decision:", decision.event_type)
    return {"event_type": decision.event_type}

def corporate_event_quote(state: CateringState):
    """Generate a quote for a corporate event."""
    prompt = ChatPromptTemplate.from_template("""
    You are a catering manager preparing a quote for a corporate event on {event_date}
    with {headcount} people, serving {menu}.
    Suggest total cost, per-person rate, and ready time, optimized for large-scale corporate catering.

    Respond in JSON:
    {{"total": number, "per_person": number, "ready_time": "HH:MM"}}
    """)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({
        "event_date": state["event_date"],
        "headcount": state["headcount"],
        "menu": ", ".join(state["menu"])
    })
    print("Corporate quote:", response)
    return {"quote": response}

def private_event_quote(state: CateringState):
    """Generate a quote for a private event."""
    prompt = ChatPromptTemplate.from_template("""
    You are a catering manager preparing a quote for a private event on {event_date}
    with {headcount} people, serving {menu}.
    Suggest total cost, per-person rate, and ready time, optimized for small-scale private catering.

    Respond in JSON:
    {{"total": number, "per_person": number, "ready_time": "HH:MM"}}
    """)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({
        "event_date": state["event_date"],
        "headcount": state["headcount"],
        "menu": ", ".join(state["menu"])
    })
    print("Private quote:", response)
    return {"quote": response}

def wedding_event_quote(state: CateringState):
    """Generate a quote for a wedding event."""
    prompt = ChatPromptTemplate.from_template("""
    You are a catering manager preparing a quote for a wedding event on {event_date}
    with {headcount} people, serving {menu}.
    Suggest total cost, per-person rate, and ready time, optimized for premium wedding catering.

    Respond in JSON:
    {{"total": number, "per_person": number, "ready_time": "HH:MM"}}
    """)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({
        "event_date": state["event_date"],
        "headcount": state["headcount"],
        "menu": ", ".join(state["menu"])
    })
    print("Wedding quote:", response)
    return {"quote": response}

def manager_gate(state: CateringState):
    """Simulated manager approval for the quote."""
    prompt = ChatPromptTemplate.from_template("""
    The catering team drafted this quote:
    {quote}

    Decide if this quote should be approved or needs revision.
    Respond JSON: 
    {{"approval_status": "approved" or "needs_revision", "reason": "<short reason>"}}
    """)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({"quote": state["quote"]})
    print("Manager review:", response)
    return {
        "approval_status": response.get("approval_status"),
        "reason": response.get("reason")
    }

def finalize(state: CateringState):
    """Return final structured output based on approval."""
    if state["approval_status"] == "approved":
        return {
            "status": "approved",
            "quote": state["quote"]
        }
    else:
        return {
            "status": "needs_revision",
            "reason": state["reason"] or "manager requested changes"
        }

# ---------------------- ROUTING LOGIC ----------------------
def route_decision(state: CateringState):
    """Route to the appropriate quote generation node based on event type."""
    if state["event_type"] == "corporate":
        return "corporate_event_quote"
    elif state["event_type"] == "private":
        return "private_event_quote"
    elif state["event_type"] == "wedding":
        return "wedding_event_quote"

# ---------------------- BUILD GRAPH ----------------------
def build_graph():
    graph = StateGraph(CateringState)

    # Add nodes
    graph.add_node("capture_request", capture_request)
    graph.add_node("route_event_type", route_event_type)
    graph.add_node("corporate_event_quote", corporate_event_quote)
    graph.add_node("private_event_quote", private_event_quote)
    graph.add_node("wedding_event_quote", wedding_event_quote)
    graph.add_node("manager_gate", manager_gate)
    graph.add_node("finalize", finalize)

    # Add edges
    graph.add_edge(START, "capture_request")
    graph.add_edge("capture_request", "route_event_type")
    graph.add_conditional_edges(
        "route_event_type",
        route_decision,
        {
            "corporate_event_quote": "corporate_event_quote",
            "private_event_quote": "private_event_quote",
            "wedding_event_quote": "wedding_event_quote"
        }
    )
    graph.add_edge("corporate_event_quote", "manager_gate")
    graph.add_edge("private_event_quote", "manager_gate")
    graph.add_edge("wedding_event_quote", "manager_gate")
    graph.add_edge("manager_gate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    app = build_graph()
    app.get_graph().print_ascii()

    # Define test catering requests
    test_requests = [
        {
            "event_date": "2025-11-12",
            "headcount": 120,
            "menu": ["grilled chicken", "pasta primavera", "salad"]
        },
        {
            "event_date": "2025-12-01",
            "headcount": 30,
            "menu": ["paneer tikka", "naan", "butter chicken"]
        },
        {
            "event_date": "2025-12-15",
            "headcount": 150,
            "menu": ["steak", "seafood risotto", "wedding cake"]
        },
        {
            "event_date": "2026-01-10",
            "headcount": 35,
            "menu": ["pasta alfredo", "garlic bread", "tiramisu"]
        },
    ]

    all_outputs = []

    print("\n Running Catering Router for Multiple Requests\n")
    for idx, req in enumerate(test_requests, start=1):
        print(f"\n=================  REQUEST #{idx} =================")
        print(json.dumps(req, indent=2))

        result = app.invoke(req)

        print("\nFinal Result:")
        print(json.dumps(result, indent=2))
        print("=====================================================")
        all_outputs.append(result)

    # Summarize all runs
    print("\nSummary of All Catering Runs:")
    for i, res in enumerate(all_outputs, start=1):
        print(f"\nRequest #{i}: {res.get('status', 'unknown').upper()}")
        if res.get("status") == "approved":
            quote = res.get("quote", {})
            print(f"   Total: {quote.get('total', 'N/A')}, "
                  f"Per-person: {quote.get('per_person', 'N/A')}, "
                  f"Ready by: {quote.get('ready_time', 'N/A')}")
        else:
            print(f"   Reason: {res.get('reason', 'no reason provided')}")