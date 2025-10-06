import random
import time
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv

# Setup
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration
MAX_RETRIES = 3
HEARTBEAT_INTERVAL = 15
HEARTBEAT_TIMEOUT = 20
MIN_TEMP_THRESHOLD = 200

# State Models
class Heartbeat(BaseModel):
    stage: str
    core_temp_c: int
    ok: bool
    timestamp: float = Field(default_factory=time.time)

class Feedback(BaseModel):
    grade: Literal["success", "failure"] = Field(description="Evaluate baking outcome")
    feedback: str = Field(description="Suggestions for improvement if failed")

class BakeState(BaseModel):
    item: str
    target_temp_c: int
    batch_size: int = 12
    current_stage: str = "init"
    stages_completed: List[str] = Field(default_factory=list)
    peak_oven_c: int = 0
    status: Literal["pending", "running", "completed", "aborted"] = "pending"
    attempt_count: int = 0
    last_heartbeat: Optional[Heartbeat] = None
    heartbeat_history: List[Heartbeat] = Field(default_factory=list)
    reason: Optional[str] = None
    start_time: float = Field(default_factory=time.time)
    decision: str = "pending"
    feedback: Optional[Feedback] = None

# Worker Subgraph
class BakeWorker:
    STAGES = ["preheat", "load", "bake", "finish"]
    STAGE_TIMES = {"preheat": 30, "load": 10, "bake": 60, "finish": 15}
    
    def __init__(self, state: BakeState):
        self.state = state
    
    def execute_stage(self, stage: str) -> bool:
        print(f"Starting {stage}")
        self.state.current_stage = stage
        stage_duration = self.STAGE_TIMES[stage]
        elapsed = 0
        
        while elapsed < stage_duration:
            time.sleep(min(HEARTBEAT_INTERVAL, stage_duration - elapsed))
            elapsed += HEARTBEAT_INTERVAL
            
            heartbeat = self._generate_heartbeat(stage)
            self.state.last_heartbeat = heartbeat
            self.state.heartbeat_history.append(heartbeat)
            print(f"Heartbeat: {stage} - {heartbeat.core_temp_c}°C - {'OK' if heartbeat.ok else 'FAULT'}")
            
            if not heartbeat.ok:
                self.state.reason = f"Fault in {stage}: temp={heartbeat.core_temp_c}°C"
                print(f"Error: {self.state.reason}")
                return False
                
            if heartbeat.core_temp_c > self.state.peak_oven_c:
                self.state.peak_oven_c = heartbeat.core_temp_c
        
        self.state.stages_completed.append(stage)
        print(f"Completed {stage}")
        return True
    
    def _generate_heartbeat(self, stage: str) -> Heartbeat:
        base_temps = {"preheat": 180, "load": 200, "bake": self.state.target_temp_c, "finish": 100}
        base_temp = base_temps[stage]
        temp_variation = random.randint(-20, 20)
        current_temp = base_temp + temp_variation
        has_fault = random.random() < 0.02 or current_temp < MIN_TEMP_THRESHOLD
        
        if has_fault:
            current_temp = random.randint(50, MIN_TEMP_THRESHOLD - 1)
            ok = False
        else:
            ok = True
            
        return Heartbeat(stage=stage, core_temp_c=current_temp, ok=ok)
    
    def run_bake(self) -> bool:
        self.state.status = "running"
        for stage in self.STAGES:
            if not self.execute_stage(stage):
                self.state.status = "aborted"
                return False
        self.state.status = "completed"
        return True

# Supervisor Logic
def initialize_bake(state: BakeState) -> BakeState:
    print(f"Initializing bake: {state.item} at {state.target_temp_c}°C")
    state.attempt_count += 1
    state.status = "pending"
    state.start_time = time.time()
    state.current_stage = "init"
    state.stages_completed = []
    state.peak_oven_c = 0
    state.last_heartbeat = None
    state.heartbeat_history = []
    state.reason = None
    return state

def execute_bake_worker(state: BakeState) -> BakeState:
    print(f"Attempt {state.attempt_count}/{MAX_RETRIES}")
    worker = BakeWorker(state)
    start_time = time.time()
    last_heartbeat_time = start_time
    
    success = worker.run_bake()
    
    if not success:
        state.status = "aborted"
        if state.last_heartbeat and state.last_heartbeat.core_temp_c < MIN_TEMP_THRESHOLD:
            state.reason = f"Temperature below {MIN_TEMP_THRESHOLD}°C"
        elif time.time() - last_heartbeat_time > HEARTBEAT_TIMEOUT:
            state.reason = "No heartbeat received"
    
    return state

def evaluate_bake(state: BakeState) -> BakeState:
    evaluator = llm.with_structured_output(Feedback)
    prompt = f"Evaluate baking {state.item} at {state.target_temp_c}°C. Status: {state.status}. Reason: {state.reason or 'None'}. Heartbeats: {'; '.join([f'{hb.stage}: {hb.core_temp_c}°C ({"OK" if hb.ok else "FAULT"})' for hb in state.heartbeat_history[-2:]])}."
    state.feedback = evaluator.invoke(prompt)
    print(f"Evaluation: {state.feedback.grade} - {state.feedback.feedback}")
    return state

def check_retry_decision(state: BakeState) -> BakeState:
    if state.status == "completed":
        state.decision = "complete"
        print("Bake successful")
    elif state.attempt_count >= MAX_RETRIES:
        state.decision = "abort"
        state.reason = state.reason or "Max retries exceeded"
        print("Aborting: Max retries reached")
    else:
        state.decision = "retry"
        print(f"Retrying with feedback: {state.feedback.feedback}")
    return state

def get_retry_decision(state: BakeState) -> str:
    return state.decision

def format_output(state: dict) -> Dict[str, Any]:
    if state["status"] == "completed":
        return {
            "status": "completed",
            "stages": state.get("stages_completed", []),
            "peak_oven_c": state.get("peak_oven_c", 0),
            "batch_size": state.get("batch_size", 0)
        }
    return {
        "status": "aborted",
        "reason": state.get("reason", "Unknown failure"),
        "attempts": state.get("attempt_count", 0)
    }

# Graph Construction
def build_supervisor_graph():
    graph = StateGraph(BakeState)
    graph.add_node("initialize", initialize_bake)
    graph.add_node("execute", execute_bake_worker)
    graph.add_node("evaluate", evaluate_bake)
    graph.add_node("check_retry", check_retry_decision)
    
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_edge("evaluate", "check_retry")
    graph.add_conditional_edges(
        "check_retry",
        get_retry_decision,
        {"complete": END, "retry": "initialize", "abort": END}
    )
    
    return graph.compile()

# Main Execution
def main():
    print("Oven Bake Supervisor")
    input_state = BakeState(item="sourdough", target_temp_c=230, batch_size=12)
    supervisor = build_supervisor_graph()
    supervisor.get_graph().print_ascii()
    result = supervisor.invoke(input_state)
    output = format_output(result)
    print("\nResult:")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()