from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI

load_dotenv()

#tools
search = TavilySearchResults(max_results=1)
tools = [search]

#agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

# Choose the LLM to use
llm = OpenAI()

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == '__main__':
    agent_executor.invoke({"input": "what is LangChain?"})






