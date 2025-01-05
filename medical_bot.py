import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache the vectorstore for efficiency
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def load_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": hf_token,
            "max_length": "1024"
        }
    )
    return llm


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    # Page settings
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 AI Chatbot Assistant")
    
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat message display
    st.subheader("Chat History")
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Input section
    prompt = st.chat_input("💬 Type your question here...")
    
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            Provide only the information within the given context.
            
            Context: {context}
            Question: {question}
            
            Start the answer directly. Avoid small talk.
        """

        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        
        try:
            # Load vectorstore
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Error: Vectorstore is not available")
                return
            
            # Build the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Get response
            response = qa_chain.invoke({'query': prompt})
            result = response['result']
            source_documents = response.get('source_documents', "No source documents available.")
            
            # Format and display response
            result_to_show = f"**Answer:** {result}\n\n**Source Documents:**\n{source_documents}"
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

