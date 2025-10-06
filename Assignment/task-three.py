from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any
import os
from dotenv import load_dotenv

# Setup
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Order State
class OrderState(BaseModel):
    order_type: Literal["dine_in", "takeout", "delivery", "unknown"] = "unknown"
    items: List[str] = Field(default_factory=list)
    address: Optional[str] = None
    requested_time: str = "ASAP"
    route: Optional[str] = None
    summary: Optional[str] = None
    prep_eta_min: Optional[int] = None
    courier_eta_min: Optional[int] = None
    notes: Optional[str] = None
    priority: Optional[str] = "normal"
    processing_status: str = "pending"
    table_number: Optional[int] = None
    special_instructions: Optional[List[str]] = Field(default_factory=list)
    total_eta_min: Optional[int] = None

# Helper Functions
def safe_llm_call(prompt_template: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | JsonOutputParser()
        return chain.invoke(input_data) or {}
    except Exception as e:
        return {"error": str(e), "fallback": True}

def calculate_priority(items: List[str], requested_time: str) -> str:
    text = f"{' '.join(items)} {requested_time}".lower()
    if any(keyword in text for keyword in ["birthday", "anniversary", "urgent", "asap", "rush", "now"]):
        return "urgent"
    if any(item in text for item in ["steak", "lobster", "celebration cake"]):
        return "high"
    return "normal"

# Route Handlers
def handle_dine_in(state: OrderState) -> OrderState:
    prompt_template = """
    Manage a dine-in order with:
    - Items: {items}
    - Requested Time: {requested_time}
    - Priority: {priority}
    - Special Instructions: {special_instructions}
    
    Return JSON:
    {{
        "route": "dine_in",
        "table_number": number,
        "prep_eta_min": number,
        "priority": "low|normal|high|urgent",
        "notes": "staff instructions"
    }}
    """
    result = safe_llm_call(prompt_template, {
        "items": ", ".join(state.items),
        "requested_time": state.requested_time,
        "priority": state.priority,
        "special_instructions": ", ".join(state.special_instructions)
    })
    
    if not result.get("fallback"):
        state.route = result.get("route", "dine_in")
        state.table_number = result.get("table_number", 1)
        state.prep_eta_min = result.get("prep_eta_min", 15)
        state.total_eta_min = state.prep_eta_min
        state.notes = result.get("notes", "Table assigned, kitchen notified.")
        state.summary = f"Dine-in: Table {state.table_number}, ETA {state.prep_eta_min}min"
        state.processing_status = "assigned"
    else:
        state.notes = "Standard dine-in processing"
        state.table_number = 1
        state.prep_eta_min = 20
        state.total_eta_min = 20
        state.summary = "Dine-in: Fallback processing"
        state.processing_status = "assigned"
    
    return state

def handle_takeout(state: OrderState) -> OrderState:
    prompt_template = """
    Manage a takeout order with:
    - Items: {items}
    - Requested Pickup: {requested_time}
    - Priority: {priority}
    - Special Instructions: {special_instructions}
    
    Return JSON:
    {{
        "route": "takeout",
        "prep_eta_min": number,
        "pickup_time": "time estimate",
        "notes": "staff instructions"
    }}
    """
    result = safe_llm_call(prompt_template, {
        "items": ", ".join(state.items),
        "requested_time": state.requested_time,
        "priority": state.priority,
        "special_instructions": ", ".join(state.special_instructions)
    })
    
    if not result.get("fallback"):
        state.route = result.get("route", "takeout")
        state.prep_eta_min = result.get("prep_eta_min", 20)
        state.total_eta_min = state.prep_eta_min
        state.notes = result.get("notes", "Takeout order queued.")
        state.summary = f"Takeout: Ready in {state.prep_eta_min}min"
        state.processing_status = "preparing"
    else:
        state.notes = "Standard takeout processing"
        state.prep_eta_min = 25
        state.total_eta_min = 25
        state.summary = "Takeout: Fallback processing"
        state.processing_status = "preparing"
    
    return state

def handle_delivery(state: OrderState) -> OrderState:
    prompt_template = """
    Coordinate a delivery order with:
    - Items: {items}
    - Address: {address}
    - Requested Time: {requested_time}
    - Priority: {priority}
    - Special Instructions: {special_instructions}
    
    Return JSON:
    {{
        "route": "delivery",
        "prep_eta_min": number,
        "courier_eta_min": number,
        "total_delivery_time": number,
        "notes": "kitchen and customer updates"
    }}
    """
    result = safe_llm_call(prompt_template, {
        "items": ", ".join(state.items),
        "address": state.address or "Address not provided",
        "requested_time": state.requested_time,
        "priority": state.priority,
        "special_instructions": ", ".join(state.special_instructions)
    })
    
    if not result.get("fallback"):
        state.route = result.get("route", "delivery")
        state.prep_eta_min = result.get("prep_eta_min", 25)
        state.courier_eta_min = result.get("courier_eta_min", 30)
        state.total_eta_min = result.get("total_delivery_time", state.prep_eta_min + state.courier_eta_min)
        state.notes = result.get("notes", "Delivery queued.")
        state.summary = f"Delivery: Total {state.total_eta_min}min"
        state.processing_status = "dispatched"
    else:
        state.notes = "Standard delivery processing"
        state.prep_eta_min = 30
        state.courier_eta_min = 35
        state.total_eta_min = 65
        state.summary = "Delivery: Fallback processing"
        state.processing_status = "dispatched"
    
    return state

def handle_unknown(state: OrderState) -> OrderState:
    prompt_template = """
    Classify an unclear order with:
    - Items: {items}
    - Address: {address}
    - Requested Time: {requested_time}
    - Special Instructions: {special_instructions}
    
    Return JSON:
    {{
        "suggested_type": "dine_in|takeout|delivery",
        "reasoning": "explanation",
        "notes": "staff instructions"
    }}
    """
    result = safe_llm_call(prompt_template, {
        "items": ", ".join(state.items),
        "address": state.address or "Not provided",
        "requested_time": state.requested_time,
        "special_instructions": ", ".join(state.special_instructions)
    })
    
    state.route = "requires_clarification"
    state.notes = result.get("notes", "Contact customer to clarify order type.")
    state.summary = f"Unknown order: {result.get('reasoning', 'needs review')}"
    state.processing_status = "pending_clarification"
    
    return state

# Router
def preprocess_order(state: OrderState) -> OrderState:
    state.priority = calculate_priority(state.items, state.requested_time)
    if state.order_type == "delivery" and not state.address:
        state.order_type = "unknown"
        state.notes = "Delivery order missing address"
    state.processing_status = "routing"
    return state

def route_order(state: OrderState) -> str:
    return {
        "dine_in": "dine_in_path",
        "takeout": "takeout_path",
        "delivery": "delivery_path",
        "unknown": "unknown_path"
    }.get(state.order_type.lower(), "unknown_path")

# Graph Construction
def build_graph():
    graph = StateGraph(OrderState)
    graph.add_node("preprocess", preprocess_order)
    graph.add_node("dine_in_path", handle_dine_in)
    graph.add_node("takeout_path", handle_takeout)
    graph.add_node("delivery_path", handle_delivery)
    graph.add_node("unknown_path", handle_unknown)
    graph.set_entry_point("preprocess")
    graph.add_conditional_edges(
        "preprocess",
        route_order,
        {
            "dine_in_path": "dine_in_path",
            "takeout_path": "takeout_path",
            "delivery_path": "delivery_path",
            "unknown_path": "unknown_path"
        }
    )
    graph.add_edge("dine_in_path", END)
    graph.add_edge("takeout_path", END)
    graph.add_edge("delivery_path", END)
    graph.add_edge("unknown_path", END)
    return graph.compile()

# Test Cases
sample_inputs = [
    OrderState(
        order_type="dine_in",
        items=["ribeye steak", "lobster tail"],
        requested_time="20:00",
        special_instructions=["table by window"]
    ),
    OrderState(
        order_type="takeout",
        items=["family combo", "kids meal"],
        requested_time="ASAP"
    ),
    OrderState(
        order_type="delivery",
        items=["large pizzas", "soda"],
        address="123 Main St",
        requested_time="within 45 minutes"
    ),
    OrderState(
        order_type="unknown",
        items=["wedding cake"],
        requested_time="next Friday"
    )
]

# Execution
if __name__ == "__main__":
    router = build_graph()
    router.get_graph().print_ascii()
    for i, order_input in enumerate(sample_inputs, 1):
        try:
            result = router.invoke(order_input)
            print(f"\nOrder #{i}: {order_input.order_type} - {', '.join(order_input.items)}")
            print(f"Route: {result.get('route', 'N/A')}")
            print(f"ETA: {result.get('total_eta_min', result.get('prep_eta_min', 'N/A'))}min")
            print(f"Summary: {result.get('summary', 'N/A')}")
            print(f"Status: {result.get('processing_status', 'N/A')}")
            if result.get('table_number'):
                print(f"Table: {result.get('table_number')}")
            print(f"Notes: {result.get('notes', 'N/A')}")
        except Exception as e:
            print(f"Error processing order #{i}: {e}")