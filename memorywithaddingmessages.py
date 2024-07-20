from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

memory = SqliteSaver.from_conn_string(":memory:")

#tools
search = TavilySearchResults(max_results=1)
tools = [search]

#agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react-chat")


# Choose the LLM to use
llm = OpenAI()

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, checkpoint=memory)
config = {"configurable": {"thread_id": "abc123"}}

if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input(">")
        chat_history.append(f"Human: {user_input}")

        response = []
        for chunk in agent_executor.stream(
                {
                    "input": user_input,
                    "chat_history": "\n".join(chat_history),
                },
                config
        ):
            if 'text' in chunk:
                print(chunk['text'], end='')
                response.append(chunk['text'])

        chat_history.append(f"AI: {''.join(response)}")
        print("\n----")








