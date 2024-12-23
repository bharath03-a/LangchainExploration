from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

# streamlit framework
st.title("Explore Your Destination!")
input_txt = st.text_input("Enter the Desitnation you want to explore")

# initializing LLAMA llm
llm = ChatOllama(
    model = "llama3.2:latest",
    temperature = 0.8
)

# langchain prompt Templates
first_prompt = PromptTemplate(
    input_variables = ['destination'],
    template = "Tell me about the destination {destination}. Include historical and cultural significance."
)

# adding converstation memory
destination_memory = ConversationBufferMemory(
    input_key = "destination",
    memory_key = "chat_history"
)

chain_1 = LLMChain(
    llm = llm,
    prompt = first_prompt,
    verbose = True,
    output_key = "description",
    memory = destination_memory
)

# second prompt template
second_prompt = PromptTemplate(
    input_variables = ["description"],
    template = "When was {description} founded or established or become famous? Share any key historical moments."
)

history_memory = ConversationBufferMemory(
    input_key = "description",
    memory_key = "dest_history"
)

chain_2 = LLMChain(
    llm = llm,
    prompt = second_prompt,
    verbose = True,
    output_key = "history",
    memory = destination_memory
)

# third prompt template
third_prompt = PromptTemplate(
    input_variables = ["history"],
    template = "List 5 must-visit attractions near {history} or historical events that shaped its identity."
)

top_memory = ConversationBufferMemory(
    input_key = "history",
    memory_key = "top_places"
)

chain_3 = LLMChain(
    llm = llm,
    prompt = third_prompt,
    verbose = True,
    output_key = "top",
    memory = top_memory
)

# creating a sequential chain
parent_chain = SequentialChain(
    chains = [chain_1, chain_2, chain_3],
    input_variables = ["destination"],
    output_variables = ["description", "history", "top"],
    verbose = True
)

if input_txt:
    st.write(parent_chain({"destination" : input_txt}))

    with st.expander("Destination Description"):
        st.info(destination_memory.buffer)

    with st.expander("Must Visit Attractions"):
        st.info(top_memory.buffer)