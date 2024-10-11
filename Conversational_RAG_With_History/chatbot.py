import streamlit as st
import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(model_name="Gemma2-9b-It")

## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

# Initialize session state to store sessions if not already done
if 'sessions' not in st.session_state:
    default_session_id = str(uuid.uuid4())
    st.session_state['sessions'] = {default_session_id: {'session_id': default_session_id, 'message_history': ChatMessageHistory()}}
    st.session_state['selected_session'] = default_session_id

# Function to add a new session
def add_new_session():
    new_session_id = str(uuid.uuid4())  # Generate unique session ID
    st.session_state['sessions'][new_session_id] = {'session_id': new_session_id, 'message_history': ChatMessageHistory()}
    st.session_state['selected_session'] = new_session_id

# Sidebar layout
st.sidebar.title("Chatbot Sessions")

# Button to add a new session
if st.sidebar.button("Add New Session"):
    add_new_session()


# Display all existing sessions in the sidebar
st.sidebar.subheader("All Sessions")
session_ids = list(st.session_state['sessions'].keys())  # Get all session IDs
selected_session = st.sidebar.selectbox("Select a Session", session_ids)

# Set the selected session in the session state
st.session_state['selected_session'] = selected_session

st.session_state['sessions'][st.session_state['selected_session']]['uploaded_files'] = st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
uploaded_files = st.session_state['sessions'][st.session_state['selected_session']]['uploaded_files']
# Process uploaded  PDF's
if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)    

    ## Answer question

    # Answer question
    system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Give a detailed description of the answer."
            "\n\n"
            "{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id in st.session_state['sessions']:
            return st.session_state['sessions'][session_id]['message_history']
    
    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    user_input = st.text_input("Your question:")
    if user_input:
        session_id = st.session_state['selected_session']
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            },  # constructs a key "abc123" in `store`.
        )
        st.write("Assistant:", response['answer'])
