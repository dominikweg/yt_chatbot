
import os
import pinecone
import streamlit as st
from urllib.parse import urlparse, parse_qs
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import  RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

YT_URL = "https://www.youtube.com/watch?v=tLS6t3FVOTI"
SAVE_DIR = 'downloads'

# Description of the tool that indicates the AI assitant when to use it
TOOL_DESC = ("""This tool is the video of the podcast episode provided to you by the user.Use this tool to answer user questions
    using video of the podcast episode by Aandrew Huberman. If the user asks about the video or podcast episode use this tool to get the answer.
    This tool also can be used for follow up questions by the user. When user asks about Andrew Huberman use this tool.
    This tool is the transcript of the video.""")

if os.path.exists(SAVE_DIR):
    print(f"Directory: {SAVE_DIR} exists")
else:
    os.mkdir(SAVE_DIR)

def get_yt_id(url):
    """Getting yt video id from url"""
    parsed_url = urlparse(url)    
    query = parse_qs(parsed_url.query)
    return query['v'][0]

def yt_id_to_chunked_docs(yt_id):
    """Splittinh the document into chunks"""
    loader = YoutubeLoader(yt_id)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunk_docs = text_splitter.split_documents(docs)
    
    return chunk_docs

def get_pinecone_db(chunk_docs, embs):
    """Initializing the vector storage"""
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
    index_name="yt-knowledge-base"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

    search = Pinecone.from_documents(chunk_docs, embs, index_name=index_name)
    
    return search

def setup_yt_knowledge_agent():
    """Setting up the conversational yt video knowledge base agent"""
    
    # Getting yt id on link change   
    yt_id = get_yt_id(YT_URL)
    
    # Loading the document and chunking it
    chunk_docs = yt_id_to_chunked_docs(yt_id)
    embs = OpenAIEmbeddings()
        
    # Getting the vector base embeddings
    search = get_pinecone_db(chunk_docs, embs)
        
    # Creating LLM conversation chain
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    
    conv_memory = ConversationBufferMemory(
        llm=llm, memory_key='chat_history', return_messages=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=search.as_retriever(),
        chain_type='stuff'
    )
    

    # Creating tool containing knowledge base about the video
    tools = [Tool(
        name='Video DB',
        func=qa_chain.run,
        description=TOOL_DESC)]
    
    # Initializing conversational agent
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm, 
        max_iterations=3, # Limiting how many times the agent can loop through the tool
        early_stopping_method='generate',
        memory=conv_memory,
        verobse=True
    )
    return agent

def app():
    
    st.set_page_config(page_title="Youtube Video Chatbot")
    st.title("Youtube Video Chatbot :clapper: :parrot: :link:")
    st.markdown("This is chatbot app based on a GPT 3.5 model which allows you to ask it questions about the YT video of your choice."+ 
                "The chosen video  is **'Developing a Rational Approach to Supplementation for Health & Performance' by Andrew Huberman**")
         
     
    st.subheader("Your YT Video Assistant")
    
    # Setting the conversation agent if not present in the session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = setup_yt_knowledge_agent()   
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
     
    # Showing the whole conversation on every rerun 
    for message in st.session_state.chat_history:     
        with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
            st.markdown(message.content)     
        
    prompt = st.chat_input("Ask your chatbot anything...")
    if prompt:
        st.chat_message("user").markdown(prompt)
            
        response = st.session_state.conversation.run(prompt)
        st.chat_message("assistant").markdown(response)
        
        # Storing the conversation to display
        st.session_state.chat_history = st.session_state.conversation.memory.chat_memory.messages


if __name__ == '__main__':
    app()