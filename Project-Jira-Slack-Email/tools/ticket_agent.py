"""Atomic Jira tools for agent-driven orchestration.

These tools perform single responsibilities and do not contain branching
logic that decides workflow paths. The LLM agent composes these tools to
decide what steps to run.
"""
import json
import logging
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from config import llm
from config import jira_client, active_workflows
from prompts.prompt import JIRA_SYSTEM_PROMPT
from dotenv import load_dotenv
load_dotenv()
import re


@tool
def create_issue(summary: str, description: str, issue_type: str = "Task") -> str:
    """Create a Jira issue and return a status that it has been created. No idempotency checks here."""
    project_key = os.getenv('JIRA_PROJECT_KEY')
    logging.info(f"TOOL: create_issue for project {project_key}")
    print("+++++++++++++++++++++++++ JIRA TOOL ++++++++++++++++++++++++++++")
    print(f"CREATE ISSUE def create_issue({summary}: str, {description}: str, {issue_type}: str = None) ")
    try:
        issue = jira_client.create_issue(
            project=project_key,
            summary=summary,
            description=description,
            issuetype={"name": issue_type},
        )
        ticket_id = issue.key  # Still capture ID for logging, but not returned
        logging.info(f"Created Issue {ticket_id}")
        return json.dumps({"status": "success", "message": "Ticket has been created"})
    except Exception as e:
        logging.error(f"Failed to create issue: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@tool
def get_issue_status(ticket_id: str) -> str:
    """Return the status name for the given ticket."""
    try:
        issue = jira_client.issue(ticket_id)
        return json.dumps({"status": str(getattr(issue.fields, 'status').name)})
    except Exception as e:
        logging.error(f"Failed to get issue status: {e}")
        return json.dumps({"status": "unknown", "error": str(e)})


@tool
def add_comment(ticket_id: str, comment: str) -> str:
    """Add a comment to the ticket (validate ticket_id first)."""
    # Basic sanity-check for Jira issue keys like PROJ-123
    if not isinstance(ticket_id, str) or not re.match(r'^[A-Z][A-Z0-9]+-\d+$', ticket_id.strip()):
        logging.warning(f"add_comment: rejecting invalid ticket_id '{ticket_id}'")
        return json.dumps({"status": "error", "message": f"invalid ticket_id '{ticket_id}'"})

    try:
        jira_client.add_comment(ticket_id, comment)
        return json.dumps({"status": "success"})
    except Exception as e:
        logging.error(f"Failed to add comment: {e}")
        return json.dumps({"status": "error", "message": str(e)})


tools = [
    create_issue,
    get_issue_status,
    add_comment,
]

Jira_Agent = create_agent(llm, tools, system_prompt=JIRA_SYSTEM_PROMPT)




# @tool
# def find_ticket_by_thread(thread_ts: str) -> str:
#     """Return the ticket id mapped to a slack thread_ts, or empty string."""
#     for tid, wf in active_workflows.items():
#         if wf.get('slack_thread_ts') == thread_ts:
#             return json.dumps({"ticket_id": tid})
#     return json.dumps({"ticket_id": ""})


# @tool
# def find_ticket_by_user_request(user_email: str, access_requested: str) -> str:
#     """Return a ticket id matching (user_email, access_requested) or empty string."""
#     for tid, wf in active_workflows.items():
#         if wf.get('user_email') == user_email and wf.get('access_requested') == access_requested:
#             return json.dumps({"ticket_id": tid})
#     return json.dumps({"ticket_id": ""})
