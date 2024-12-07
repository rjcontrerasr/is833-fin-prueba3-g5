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
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
st.write(url)

# Load the dataset if a valid URL is provided
if url:
    try:
        df1 = pd.read_csv(url)
        st.write(df1)
    except Exception as e:
        st.error(f"An error occurred: {e}")

product_categories = df1['Product'].unique().tolist()

### Important part.
# Create a session state variable to flag whether the app has been initialized.
# This code will only be run the first time the app is loaded.
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"

    # Initialize the memory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True)

    # LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Tools
    from langchain.agents import tool
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date, use this for any \
        questions that need today's date to be answered. \
        This tool returns a string with today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]
    
    # Create the agent with memory
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
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Define a key in session state to store the identified product
if "identified_product" not in st.session_state:
    st.session_state.identified_product = None

# Display the existing chat messages via `st.chat_message`
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)

# Create a chat input field to allow the user to enter a message
if prompt := st.chat_input("How can I help?"):
    
    # User message
    st.chat_message("user").write(prompt)

    # Generate a response using the OpenAI API
    response = st.session_state.agent_executor.invoke({"input": prompt})['output']
    
    # Extract the identified product category from the response
    identified_product = None
    for category in product_categories:
        if category.lower() in response.lower():
            identified_product = category
            st.session_state.identified_product = category
            break

    # Create a single unified response message
    if identified_product:
        unified_response = (
            f"Thank you for providing the details of your issue. Based on your description, your complaint has been categorized under: **{identified_product}**. "
            "A ticket has been created for your issue, and it will be forwarded to the appropriate support team. They will reach out to you shortly to assist you further. "
            "If you have any more questions or need additional assistance, please let me know!"
        )
        st.chat_message("assistant").write(unified_response)

        # Filter the dataset to find subcategories for the identified product
        subproducts = df1[df1['Product'] == identified_product]['Sub-product'].unique().tolist()

        # Display the subproducts list
        st.write(f"Subcategories for the product category **{identified_product}**:")
        st.write(subproducts)
    else:
        st.chat_message("assistant").write(response)  # Default response when no category is identified

# Display the stored product category, if any
if st.session_state.identified_product:
    st.sidebar.write(f"Stored Product: {st.session_state.identified_product}")
