from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Fixed: Use available model

# ---------------------- STATE MODEL -----------------------
class RestaurantState(BaseModel):
    service_area: str
    inventory: Optional[Dict[str, str]] = None
    floor: Optional[Dict[str, int]] = None
    delivery: Optional[Dict[str, int]] = None
    slogan: Optional[str] = None
    review: Optional[str] = None
    tip: Optional[str] = None
    creative_summary: Optional[str] = None
    overall: Optional[str] = None
    combined_output: Optional[Dict] = None

# ---------------------- NODE FUNCTIONS -----------------------
def check_inventory(state: RestaurantState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    You are monitoring the restaurant inventory in the {area} service area.
    Give a quick JSON summary of stock levels for:
    - steak
    - pasta
    - lettuce
    Respond strictly in JSON:
    {{"steak": "...", "pasta": "...", "lettuce": "..."}}
    Use values like "low", "ok", or "critical".
    """)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"area": state.service_area})
    return {"inventory": result}

def check_floor(state: RestaurantState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    You are analyzing the floor occupancy for a restaurant in {area}.
    Estimate how many tables are open, and how long the waitlist might be.
    Return JSON:
    {{"open_tables": number, "waitlist": number}}
    """)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"area": state.service_area})
    return {"floor": result}

def check_delivery(state: RestaurantState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    You are monitoring delivery drivers in {area}.
    Estimate the number of drivers currently active and the average ETA (in minutes).
    Return JSON:
    {{"drivers_on_duty": number, "avg_eta_min": number}}
    """)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"area": state.service_area})
    return {"delivery": result}

def generate_slogan(state: RestaurantState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    Create a catchy slogan for a restaurant in the {area} service area.
    Keep it short and relevant to the area's vibe.
    """)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"area": state.service_area})
    return {"slogan": result}

def generate_review(state: RestaurantState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    Write a brief customer review (2-3 sentences) for a restaurant in the {area} service area.
    Reflect the area's atmosphere and dining experience.
    """)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"area": state.service_area})
    return {"review": result}

def generate_tip(state: RestaurantState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    Provide a helpful dining tip for customers visiting a restaurant in the {area} service area.
    Keep it concise and relevant to the area's dining scene.
    """)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"area": state.service_area})
    return {"tip": result}

def merge_results(state: RestaurantState) -> Dict:
    # Calculate overall status
    try:
        waitlist = state.floor.get("waitlist", 0) if state.floor else 0
        avg_eta = state.delivery.get("avg_eta_min", 0) if state.delivery else 0
        busy_score = waitlist + avg_eta
        steak_status = state.inventory.get("steak", "ok") if state.inventory else "ok"
        overall = "busy" if busy_score > 35 or steak_status == "low" else "steady"
    except Exception:
        overall = "unknown"

    # Combine creative outputs into a summary
    creative_summary = (
        f"Creative Snapshot for {state.service_area} Restaurant:\n\n"
        f"SLOGAN:\n{state.slogan or 'N/A'}\n\n"
        f"REVIEW:\n{state.review or 'N/A'}\n\n"
        f"TIP:\n{state.tip or 'N/A'}"
    )

    # Create combined JSON output
    combined_output = {
        "service_area": state.service_area,
        "inventory": state.inventory or {},
        "floor": state.floor or {},
        "delivery": state.delivery or {},
        "slogan": state.slogan or "",
        "review": state.review or "",
        "tip": state.tip or "",
        "overall": overall
    }

    return {
        "overall": overall,
        "creative_summary": creative_summary,
        "combined_output": combined_output
    }

# ---------------------- BUILD GRAPH ------------------------
def build_graph():
    graph = StateGraph(RestaurantState)
    
    # Add nodes for each task
    graph.add_node("check_inventory", check_inventory)
    graph.add_node("check_floor", check_floor)
    graph.add_node("check_delivery", check_delivery)
    graph.add_node("generate_slogan", generate_slogan)
    graph.add_node("generate_review", generate_review)
    graph.add_node("generate_tip", generate_tip)
    graph.add_node("merge_results", merge_results)
    
    # Add edges for parallel execution
    graph.add_edge(START, "check_inventory")
    graph.add_edge(START, "check_floor")
    graph.add_edge(START, "check_delivery")
    graph.add_edge(START, "generate_slogan")
    graph.add_edge(START, "generate_review")
    graph.add_edge(START, "generate_tip")
    
    # All tasks must complete before merging
    graph.add_edge("check_inventory", "merge_results")
    graph.add_edge("check_floor", "merge_results")
    graph.add_edge("check_delivery", "merge_results")
    graph.add_edge("generate_slogan", "merge_results")
    graph.add_edge("generate_review", "merge_results")
    graph.add_edge("generate_tip", "merge_results")
    
    # End after merging
    graph.add_edge("merge_results", END)
    
    return graph.compile()

# ---------------------- MAIN ------------------------
if __name__ == "__main__":
    app = build_graph()
    app.get_graph().print_ascii()
    input_state = RestaurantState(service_area="downtown")
    
    final_state = app.invoke(input_state)
    
    print("\n AI-Driven Dinner Rush Snapshot:")
    # Fixed: Access as dictionary keys instead of object attributes
    print(json.dumps(final_state.get("combined_output", {}), indent=2))
    print("\n Creative Summary:")
    print(final_state.get("creative_summary", "No creative summary available"))