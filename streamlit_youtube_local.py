# streamlit_youtube_local.py

import streamlit as st
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import wikipedia

# Streamlit UI
st.title("🦜🔗 Local YouTube GPT Creator (Free & Offline)")
prompt = st.text_input("Enter a video topic...")

# Prompt Templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a catchy YouTube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="Write a full YouTube video script based on this title:\n\nTitle: {title}\n\nInclude relevant facts from this Wikipedia research:\n{wikipedia_research}"
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Load local LLM with ctransformers
llm = CTransformers(
    model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # replace with your path if needed
    model_type="mistral",
    config={'max_new_tokens': 512, 'temperature': 0.7}
)

# LangChain chains
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script", memory=script_memory)

# Wikipedia wrapper (simplified)
def get_wikipedia_summary(topic):
    try:
        return wikipedia.summary(topic, sentences=5)
    except:
        return "No Wikipedia information found."

# Main logic
if prompt:
    title = title_chain.run(prompt)
    wiki_research = get_wikipedia_summary(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.subheader("📺 Video Title")
    st.write(title)

    st.subheader("📝 Video Script")
    st.write(script)

    with st.expander("🧠 Title Memory"):
        st.info(title_memory.buffer)

    with st.expander("🧠 Script Memory"):
        st.info(script_memory.buffer)

    with st.expander("📚 Wikipedia Research"):
        st.info(wiki_research)
