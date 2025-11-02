# /handlers/slack_events.py
import logging
import threading
from langchain_core.messages import HumanMessage

from config import slack_app # Import from central config
from graph.agent import workflow # Import the compiled graph

@slack_app.event("app_mention")
def handle_app_mention(body, say):
    """
    This function is triggered when the Slack bot is @mentioned.
    It parses the user's request, constructs an initial prompt,
    and invokes the agent graph in a new thread.
    """
    event = body["event"]
    user_text = event.get("text", "")
    thread_ts = event.get("thread_ts", event["ts"])
    user_id = event.get("user")
    channel = event["channel"]

    try:
        
        user_info = slack_app.client.users_info(user=user_id)
        user_profile = user_info["user"]["profile"]
        
        requester_email = user_profile.get("email")
        if not requester_email:
            logging.warning(f"No email found for Slack user {user_id}, using placeholder: {requester_email}")

    
        prompt = f"""
        A user has requested IT access. Your goal is to ensure the request is acknowledged in Slack and a Jira ticket is created.

        - User's Request: "{user_text}"
        - Requester's Email: yashjoshi1485@gmail.com
        - Slack Context: channel='{channel}', thread_ts='{thread_ts}'

        First, acknowledge the user's request in Slack. Then, create a Jira ticket.
        After you create the Jira ticket, notify in the Slack channel about the Jira ticket—mention the Jira ticket ID also.
        """
        initial_messages = [HumanMessage(content=prompt)]

        config = {"configurable": {"thread_id": f"slack-{thread_ts}"}}

        initial_state = {
            "messages": initial_messages, 
            'completed_step' : ["Starting Orchestration Process"]
        }
        threading.Thread(target=workflow.invoke, args=(initial_state, config)).start()

    except Exception as e:
        logging.error(f"Error handling app mention: {e}", exc_info=True)
        say(f"An error occurred while processing your request: {e}", thread_ts=thread_ts)