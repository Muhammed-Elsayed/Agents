# pip install -r requirment.txt
import os
from pydantic import BaseModel
from langchain_core.utils.uuid import uuid7

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, HumanMessage


load_dotenv()

# Pydantic model.
class Answer(BaseModel):
    summary: str
    confidence: float

# State of the agents (Conversation history).
class MyState(AgentState):
    user_id: str
    call_count: int

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatOpenAI(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    use_responses_api=True,
)
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    response_format=Answer,
    state_schema=MyState,

)

config = {"configurable": {"thread_id": str(uuid7())}}

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]},
    version="v3",
)
for snapshot in stream.values:
    # Each snapshot contains the full state at that point
    latest_message = snapshot["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
