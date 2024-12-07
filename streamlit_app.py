import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
import pandas as pd
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain import hub

# Show title and description
st.title("💬 Financial Support Chatbot")

### Adding subproducts

# Add a text input field for the GitHub raw URL
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
st.write(url)

# Load the dataset if a valid URL is provided
if url:
    try:
        df1 = pd.read_csv(url)
    except Exception as e:
        st.error(f"An error occurred: {e}")

product_categories = df1['Product'].unique().tolist()

### Initialization
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"
    
    # Initialize memory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True)
    
    # Initialize variables
    st.session_state.problem_described = False
    st.session_state.product_described = None
    st.session_state.jira_task_created = False
    
    # LLM and tools setup
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)
    
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. Wait for the user to describe their problem. Once the issue is described, classify the complaint strictly based on these possible categories: {product_categories}. Kindly inform the user that a ticket has been created, provide the category assigned to their complaint, and reassure them that the issue will be forwarded to the appropriate support team, who will reach out to them shortly. Maintain a professional and empathetic tone throughout."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Helper function to extract product
def extract_product_from_input(user_input, product_list):
    """
    Extract the product from user input based on predefined product categories.
    """
    for product in product_list:
        if product.lower() in user_input.lower():
            return product
    return None

# Helper function to update problem description status
def update_problem_flag(user_input):
    """
    Check if the user input is sufficient to describe a problem.
    """
    if len(user_input) > 10:  # Check if the input is detailed enough
        st.session_state.problem_described = True
        product = extract_product_from_input(user_input, product_categories)
        if product:
            st.session_state.product_described = product
            return f"Thank you for describing your issue. We detected the product: {product}. We will now proceed to create a task."
        else:
            return "Thank you for describing your issue. However, we couldn't detect a product. Could you specify the product?"
    else:
        return "Could you please provide more details about your issue?"

# Display existing chat messages
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)

# Chat input and problem/product handling
if prompt := st.chat_input("How can I help?"):
    st.chat_message("user").write(prompt)
    
    if not st.session_state.problem_described:
        response = update_problem_flag(prompt)  # Update the problem flag and check for product
    else:
        # Generate a response from the agent if the problem is already described
        response = st.session_state.agent_executor.invoke({"input": prompt})['output']
    
    st.chat_message("assistant").write(response)

# Conditional Jira task creation
if st.session_state.problem_described and st.session_state.product_described and not st.session_state.jira_task_created:
    st.session_state.jira_task_created = True  # Prevent duplicate task creation

    @tool
    def datetoday(dummy: str) -> str:
    """Returns today's date."""
    return "Today is " + str(date.today())
    
    # Setup Jira API
    os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
    os.environ["JIRA_USERNAME"] = "rich@bu.edu"
    os.environ["JIRA_INSTANCE_URL"] = "https://is883-genai-r.atlassian.net/"
    os.environ["JIRA_CLOUD"] = "True"
    
    assigned_issue = f"Managing my {st.session_state.product_described} Account"
    client_complaint = prompt  # Use the user's latest input as the complaint description
    
    question = (
        f"Create a task in my project with the key FST. Take into account that the Key of this project is FST. "
        f"The task's type is 'Task', assigned to rich@bu.edu. "
        f"The summary is '{assigned_issue}'. "
        f"Always assign 'Highest' priority if the issue is related to fraudulent activities. "
        f"Use 'High' priority for other issues. "
        f"The description is '{client_complaint}'."
    )
        
    # Create Jira task
    try:
        jira = JiraAPIWrapper()
        toolkit = JiraToolkit.from_jira_api_wrapper(jira)
        
        # Fix tool names and descriptions
        for idx, tool in enumerate(toolkit.tools):
            toolkit.tools[idx].name = toolkit.tools[idx].name.replace(" ", "_")
            if "create_issue" in toolkit.tools[idx].name:
                toolkit.tools[idx].description += " Ensure to specify the project ID."
        
        tools = toolkit.get_tools()
        chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model="gpt-4o-mini")
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(chat, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        result = agent_executor.invoke({"input": question})
        st.success(f"Jira task created successfully for the product: {st.session_state.product_described}")
    except Exception as e:
        st.error(f"Error during Jira task creation: {e}")
