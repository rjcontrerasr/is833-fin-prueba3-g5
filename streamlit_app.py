import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub
import pandas as pd
import os

# Show title and description.
st.title("💬 Financial Support Chatbot")

### Adding subproducts


# Add a text input field for the GitHub raw URL
#url = st.text_input("Enter the GitHub raw URL of the CSV file:", "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv")
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"

st.write(url)

# Load the dataset if a valid URL is provided
if url:
    try:
        df1 = pd.read_csv(url)
        #st.write("CSV Data:")
        #st.write(df1)
    except Exception as e:
        st.error(f"An error occurred: {e}")

product_categories = df1['Product'].unique().tolist()


### Important part.
# Create a session state variable to flag whether the app has been initialized.
# This code will only be run first time the app is loaded.
if "memory" not in st.session_state: ### IMPORTANT.
    model_type="gpt-4o-mini"

    # initialize the momory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True) ### IMPORTANT to use st.session_state.memory.

    # LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # tools
    from langchain.agents import tool
    from datetime import date
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date, use this for any \
        questions that need today's date to be answered. \
        This tool returns a string with today's date.""" #This is the desciption the agent uses to determine whether to use the time tool.
        return "Today is " + str(date.today())

    tools = [datetoday]
    
    # Now we add the memory object to the agent executor
    # prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(chat, tools, prompt)
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. Wait for the user to describe their problem. Once the issue is described, classify the complaint strictly based on these possible categories: {product_categories}. Kindly inform the user that a ticket has been created, provide the category assigned to their complaint, and reassure them that the issue will be forwarded to the appropriate support team, who will reach out to them shortly. Maintain a professional and empathetic tone throughout."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools,  memory=st.session_state.memory, verbose= True)  # ### IMPORTANT to use st.session_state.memory and st.session_state.agent_executor.


# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.memory.buffer:
    # if (message.type in ["ai", "human"]):
    st.chat_message(message.type).write(message.content)

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("How can I help?"):
    
    # question
    st.chat_message("user").write(prompt)

    # Generate a response using the OpenAI API.
    response = st.session_state.agent_executor.invoke({"input": prompt})['output']
    
    # response
    st.chat_message("assistant").write(response)
    # st.write(st.session_state.memory.buffer)



os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
os.environ["JIRA_USERNAME"] = "rich@bu.edu"
os.environ["JIRA_INSTANCE_URL"] = "https://is883-genai-r.atlassian.net/"
os.environ["JIRA_CLOUD"] = "True"


assigned_issue= "Managing my f Account"
client_complaint = "I made a purchase and it was disputed"

question = (
    f"Create a task in my project with the key FST. Take into account tha the Key of this project is FST "
    f"The task's type is 'Task', assignee to rich@bu.edu,"
    f"The summary is '{assigned_issue}'."
    #f"with the priority '{priority}' and the description '{client_complaint}'. "
    f"Always assign 'Highest' priority if the '{assigned_issue}' is related to fraudulent activities. Fraudulent activities include terms or contexts like unauthorized access, theft, phishing, or stolen accounts. Be strict in interpreting fraud-related issues."
    f"with the priority 'High' for other type of issues"
    f"with the description '{client_complaint}'. "
    #f"with a status  'TO DO'. "
)

#agent_executor.invoke({"input": question}, handle_parsing_errors=True)

# Execute the agent to create the Jira task

# Initialize Jira API Wrapper and Toolkit
jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)

# Fix tool names and descriptions in the toolkit
for idx, tool in enumerate(toolkit.tools):
    toolkit.tools[idx].name = toolkit.tools[idx].name.replace(" ", "_")
    if "create_issue" in toolkit.tools[idx].name:
        toolkit.tools[idx].description += " Ensure to specify the project ID."

# Add tools for the agent
tools = toolkit.get_tools()

# LLM Setup for LangChain
#chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model="gpt-4o-mini")

# Prepare the LangChain ReAct Agent
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create a question prompt for the Jira task
# priority = "Highest" if "fraud" in client_complaint.lower() else "High"


try:
    result = agent_executor.invoke({"input": question})
    print("Agent Output:", result)
except Exception as e:
    print(f"Error during Jira task creation: {e}")


