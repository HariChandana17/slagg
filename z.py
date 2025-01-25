import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def create_vector_embeddings(uploaded_files):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_docs = []
    document_names = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        try:
            if file_extension == ".pdf":
                with open(uploaded_file.name, "wb") as temp_pdf:
                    temp_pdf.write(uploaded_file.read())
                loader = PyPDFLoader(uploaded_file.name)
            elif file_extension in (".txt", ".md"):
                loader = TextLoader(uploaded_file)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                continue

            documents = loader.load()
            all_docs.extend(documents)
            document_names.extend([doc.metadata['source'] for doc in documents])

        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)
    vectors = FAISS.from_documents(texts, embeddings)
    return vectors, document_names

def document_retrieval_tab():
    st.title("📄 DocuMind")
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Create Vector Store"):
            with st.spinner("Creating vector store..."):
                vectors, document_names = create_vector_embeddings(uploaded_files)
                st.success("Vector Store DB is ready!")
                st.session_state["vectors"] = vectors
                st.session_state["document_names"] = document_names

        if "vectors" in st.session_state:
            query = st.text_input("Enter your query:")
            if query:
                with st.spinner("Retrieving answer..."):
                    try:
                        llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name="Llama3-8b-8192")
                        prompt = ChatPromptTemplate.from_template(
                            """Answer the questions based on the provided context only.
                            Please provide the most accurate response based on the question.
                            <context>
                            {context}
                            <context>
                            Questions: {input}"""
                        )
                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retrieval_chain = create_retrieval_chain(
                            st.session_state["vectors"].as_retriever(), document_chain
                        )

                        response = retrieval_chain.invoke({'input': query})
                        st.subheader("Answer:")
                        st.write(response['answer'])

                        st.subheader("Relevant Documents:")
                        for i, doc in enumerate(response["context"]):
                            if i < len(st.session_state["document_names"]):
                                document_name = st.session_state["document_names"][i]
                                with st.expander(f"Document {i+1} - {document_name}"):
                                    st.write(doc.page_content)
                            else:
                                st.warning("Mismatch between document context and names.")

                    except Exception as e:
                        st.error(f"Error retrieving documents: {str(e)}")

# Page Configuration
st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main App
document_retrieval_tab()