"""
LangGraph Agent for Course Q&A with Email Integration
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import os
from course_data import get_course_context, COURSE_INFO


# Initialize the Gmail service (will be set from app.py)
gmail_service = None


def set_gmail_service(service):
    """Set the Gmail service for the agent to use"""
    global gmail_service
    gmail_service = service


@tool
def send_email_to_professor(question: str, attachments_info: str = "") -> str:
    """
    Send an email to the professor with a student's question.
    
    Args:
        question: The student's question or message to send to the professor
        attachments_info: Information about any attachments (file names)
    
    Returns:
        A confirmation message about the email being sent
    """
    from helper_functions.gmail_send import send_email
    
    if not gmail_service:
        return "Error: Email service not initialized. Please check your Gmail authentication."
    
    try:
        subject = f"Question about {COURSE_INFO['course_code']} - {COURSE_INFO['course_name']}"
        
        body = f"""Dear Professor {COURSE_INFO['professor']['name']},

I have a question regarding the course:

{question}

"""
        if attachments_info:
            body += f"\nAttached files: {attachments_info}\n"
        
        body += """
Thank you for your time and assistance.

Best regards,
A Student"""
        
        # Get attachment paths from session state if available
        attachment_paths = None
        if hasattr(send_email_to_professor, 'attachment_paths'):
            attachment_paths = send_email_to_professor.attachment_paths
        
        result = send_email(
            service=gmail_service,
            to=COURSE_INFO['professor']['email'],
            subject=subject,
            body=body,
            attachment_paths=attachment_paths
        )
        
        return f"✅ Email successfully sent to Professor {COURSE_INFO['professor']['name']} at {COURSE_INFO['professor']['email']}"
    
    except Exception as e:
        return f"❌ Error sending email: {str(e)}"


# Define the agent state
class AgentState(TypedDict):
    messages: list
    course_context: str


def create_agent():
    """Create and return the LangGraph agent"""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to LLM
    tools = [send_email_to_professor]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Define the agent node
    def agent_node(state: AgentState):
        course_context = get_course_context()
        
        system_message = SystemMessage(content=f"""You are a helpful course assistant for {COURSE_INFO['course_code']} - {COURSE_INFO['course_name']}.

Your role is to:
1. Answer student questions about the course using the provided course information
2. If you cannot answer a question with the available course information, offer to send an email to the professor

COURSE CONTEXT:
{course_context}

IMPORTANT INSTRUCTIONS:
- Always try to answer from the course context first
- Be friendly, clear, and concise
- If information is not in the course context, politely say you don't have that information and offer to email the professor
- When offering to email the professor, ask the student to confirm
- Format dates and information clearly
- If a student explicitly asks to email the professor or agrees to send an email, use the send_email_to_professor tool

RESPONSE GUIDELINES:
- For grading questions: provide the grading breakdown or scale
- For dates: provide specific dates from the important dates section
- For policies: cite the relevant policy
- For assignments: provide assignment details and due dates
- For office hours: provide professor's office hours and location
""")
        
        messages = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    # Define routing function
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def run_agent(user_message: str, uploaded_files: list = None):
    """
    Run the agent with a user message
    
    Args:
        user_message: The user's question or message
        uploaded_files: List of file paths for attachments
    
    Returns:
        The agent's response
    """
    # Set attachment paths if files are uploaded
    if uploaded_files:
        send_email_to_professor.attachment_paths = uploaded_files
        attachments_info = ", ".join([os.path.basename(f) for f in uploaded_files])
        # Add attachment info to the user message
        user_message += f"\n[Note: User has attached the following files: {attachments_info}]"
    else:
        send_email_to_professor.attachment_paths = None
    
    app = create_agent()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "course_context": get_course_context()
    }
    
    # Run the agent
    result = app.invoke(initial_state)
    
    # Extract the final response
    final_message = result["messages"][-1]
    
    if hasattr(final_message, 'content'):
        return final_message.content
    else:
        return str(final_message)