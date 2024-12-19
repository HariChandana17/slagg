import streamlit as st
import google.generativeai as genai
from datetime import datetime
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain.embeddings import GooglePalmEmbeddings
import time
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import hashlib
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import random
from datetime import timedelta
import re
import requests
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
import base64
from phi.agent import Agent  # For video analysis
from phi.model.google import Gemini  # For video analysis
from phi.tools.duckduckgo import DuckDuckGo # For video analysis
from google.generativeai import upload_file, get_file # For video analysis
import tempfile # For video analysis
import pathlib # For video analysis
from googleapiclient.discovery import build
from scholarly import scholarly

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
print(os.getenv("GOOGLE_API_KEY"))
os.environ['SERPAPI_KEY'] = os.getenv("GOOGLE_API_KEY")

course_data = [
    {"title": "Python for Beginners", "duration": "10 hours", "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSnfKJoBOABrsOsPGF1h4PS5FHkdhlsKNcOlw&s", "url": "https://www.youtube.com/watch?v=your_python_video_link"},
    {"title": "Data Science with Python", "duration": "20 hours", "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHaFQIkBGXeUMZ28MaJYSvbp3zF7R08bypGw&s", "url": "https://www.youtube.com/watch?v=your_data_science_video_link"},
    {"title": "Machine Learning Fundamentals", "duration": "15 hours", "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLmYFMnFsCHqfRkgBhEzNpYTsyIci-LUYg4A&s", "url": "https://www.youtube.com/watch?v=your_ml_video_link"},
    {"title": "Generative AI for Beginners", "duration": "10 hours", "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT4uNKS4TQDvLCM58dq65UM3q6J-8S-YLXFmQ&s", "url": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/generative-ai-for-beginners---a-12-lesson-course/3968583"}
    # Add more courses here, each with a 'thumbnail' and 'url'
]

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
    
def gemini_pro_chat_tab():
    st.title("🤖 Gemini - Ask me anything")

    # Initialize chat session
    if "gemini_pro_chat_session" not in st.session_state:
        model = genai.GenerativeModel('gemini-pro')  # Initialize inside the function
        st.session_state.gemini_pro_chat_session = model.start_chat(history=[])

    chat_session = st.session_state.gemini_pro_chat_session

    # Display chat history
    for message in chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # User input
    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        gemini_response = chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

    

def display_courses(courses):
    for course in courses:
        col1, col2 = st.columns([1, 3])
        with col1:
            if 'thumbnail' in course and course['thumbnail']:
                try:
                    if course['thumbnail'].startswith("http://") or course['thumbnail'].startswith("https://"):
                        st.image(course['thumbnail'])
            
                    else:
                        st.error(f"Image file not found: ")
                except Exception as e:
                    st.error(f"Error loading thumbnail: ")
            else:
                st.write("No thumbnail available.")

        with col2:
            st.subheader(course['title'])
            st.write(f"Duration: {course['duration']}")
            if 'url' in course and course['url']:
                try:
                    # Make the course title a clickable link
                    st.markdown(f"[Go to Course]({course['url']})")
                except Exception as e:
                    st.error(f"Error creating course link: ")
            else:
                st.write("No course link available.")


def courses_tab():
    st.title("📚 Courses")
    display_courses(course_data)

def video_analyzer_tab():
    st.title("🎬 Media Mind")
    
    # Initialize single agent with both capabilities
    @st.cache_resource
    def initialize_agent():
        return Agent(
            name="Multimodal Analyst",
            model=Gemini(id="gemini-2.0-flash-exp"),  # Use appropriate Gemini model ID
            tools=[DuckDuckGo()],
            markdown=True,
        )

    agent = initialize_agent()

    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])

    if uploaded_file:
        # Ensure temp directory exists
        temp_dir = pathlib.Path(tempfile.gettempdir()) / "video_analyzer"
        temp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.video(video_path)
        
        user_prompt = st.text_area(
            "What would you like to know?",
            placeholder="Ask any question related to the video...",
            help="Ask about the video content and get information from the web."
        )
        
        if st.button("Analyze & Research"):
            if not user_prompt:
                st.warning("Please enter your question.")
            else:
                try:
                    with st.spinner("Processing video and researching..."):
                        video_file = upload_file(video_path)  # Ensure video_path is accessible
                        while video_file.state.name == "PROCESSING":
                            time.sleep(2)
                            video_file = get_file(video_file.name)

                        prompt = f"""
                        You are a helpful chatbot. Analyze the video and answer the user's query. 
                        User query: {user_prompt}
                        """
                        
                        result = agent.run(prompt, videos=[video_file])
                        
                    st.subheader("Result")
                    st.markdown(result.content)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up the temporary file.  This is crucial.
                    Path(video_path).unlink()


    else:
        st.info("Please upload a video to begin analysis.")

def create_vector_embeddings(uploaded_files):
    
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_docs = []
    document_names = []  # Initialize list to store document names

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            if file_extension == ".pdf":
                # For PDFs, we still need the temporary file approach if they are in memory
                with open(uploaded_file.name, "wb") as temp_pdf:
                    temp_pdf.write(uploaded_file.read()) #Use tempfile if PDF is purely in memory
                loader = PyPDFLoader(uploaded_file.name) #Load temp file, not the original upload
                
            elif file_extension in (".txt", ".md"):
                loader = TextLoader(uploaded_file)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                continue

            documents = loader.load()
            all_docs.extend(documents)
            document_names.extend([doc.metadata['source'] for doc in documents])  # Get document names

        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {e}")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)

    vectors = FAISS.from_documents(texts, embeddings)
    return vectors, document_names


# Function for the Document Retrieval tab
def document_retrieval_tab():
    st.title("📄 DocuMind")

    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Create Vector Store"):
            with st.spinner("Creating vector store..."):
                vectors, document_names = create_vector_embeddings(uploaded_files)
                st.success("Vector Store DB is ready!")

                # Store vectors and names in session state
                st.session_state["vectors"] = vectors
                st.session_state["document_names"] = document_names

        if "vectors" in st.session_state:  # Check if vectors are available
            query = st.text_input("Enter your query:")
            if query:
                with st.spinner("Retrieving answer..."):
                    try:
                        # Initialize LLM and chain (replace with your actual API key and model)
                        llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name="Llama3-8b-8192") # Replace with your LLM if needed
                        prompt = ChatPromptTemplate.from_template(
                            """Answer the questions based on the provided context only.
                            Please provide the most accurate response based on the question.
                            <context>
                            {context}
                            <context>
                            Questions: {input}"""
                        )
                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retrieval_chain = create_retrieval_chain(st.session_state["vectors"].as_retriever(), document_chain)


                        start_time = time.process_time()
                        response = retrieval_chain.invoke({'input': query})
                        response_time = time.process_time() - start_time

                        st.write(f"Response time: {response_time:.2f} seconds")
                        st.subheader("Answer:")
                        st.write(response['answer'])


                        # Display relevant documents with names
                        st.subheader("Relevant Documents:")
                        for i, doc in enumerate(response["context"]):
                            if i < len(st.session_state["document_names"]):  # Check index bounds
                                document_name = st.session_state["document_names"][i]
                                with st.expander(f"Document {i+1} - {document_name if document_name else 'Unnamed Document'}"):
                                    st.write(doc.page_content)
                            else:
                                st.warning("Mismatch between document context and names.")

                    except Exception as e:
                        st.error(f"Error retrieving documents: {str(e)}")


# Database setup
def init_db():
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            user_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_user_data(username, data):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('UPDATE users SET user_data = ? WHERE username = ?', 
              (json.dumps(data), username))
    conn.commit()
    conn.close()

def load_user_data(username):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('SELECT user_data FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0]:
        return json.loads(result[0])
    return None

# Initialize database
init_db()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Set up Google Gemini API configuration
genai.configure(api_key=GEMINI_API_KEY)

# Model Configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Page Configuration
st.set_page_config(
    page_title="Smart Learning Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication functions
def create_user(username, password):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, user_data) VALUES (?, ?, ?)',
                 (username, hash_password(password), json.dumps({
                     'learning_paths': {},
                     'study_time': {},
                     'achievements': [],
                     'goals': [],
                     'learning_style': None
                 })))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

# Login/Register UI
def show_login_page():
    st.title("🎓 Smart Learning Hub")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if verify_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state.user_data = load_user_data(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif create_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

from googleapiclient.discovery import build
import requests

# YouTube Data API Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube_service = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# SERPAPI Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_BASE_URL = "https://serpapi.com/search.json"

# def generate_learning_path(target, learning_style, goal_type):
#     # Simplified learning path generation
#     path_prompt = f"""
#     Create a concise learning path for {target} optimized for {learning_style['style']} learners.
#     Provide 5-6 key learning stages with specific, actionable topics.
#     Format as a simple, clear list of key learning milestones.
#     """
#     try:
#         # Generate core learning path
#         path_response = model.generate_content(path_prompt)
#         path_content = path_response.text.strip()

#         # Split into topics
#         #topics = [item.strip() for item in path_content.split('\n') if item.strip() and item.strip().startswith(('1.', '2.', '•', '-'))]
#         #if not topics:
#         #    topics = [item.strip() for item in path_content.split('\n') if item.strip()][:6]
#         topics = [item.strip() for item in path_content.split('\n') if item.strip()]
        
#         resources = {'steps': []}

#         # Scholarly and YouTube resources initialization
#         for topic in topics:
#             step_resources = {
#                 'topic': topic,
#                 'recommended_resources': [],
#                 'scholarly_articles': [],
#                 'youtube_videos': []
#             }

#             # Generate basic resources for each topic
#             resource_prompt = f"""
#             Suggest 3 key learning resources for the topic: {topic}
#             Include:
#             - 1 online course or tutorial
#             - 1 book or reading resource
#             - 1 practical project or exercise
#             """
#             resource_response = model.generate_content(resource_prompt)
#             resources_text = resource_response.text.strip()

#             # Parse resources
#             step_resources['recommended_resources'] = [
#                 resource.strip() for resource in resources_text.split('\n') 
#                 if resource.strip() and not resource.strip().lower().startswith(('topic', 'suggestion'))
#                 ]
#             # Scholarly articles extraction
#             scholar_articles = scholarly.search_pubs(f"{target} {topic}")
#             step_resources['scholarly_articles'] = [
#                     {
#                         'title': next(scholar_articles).get('bib', {}).get('title', 'Untitled'),
#                         'authors': ', '.join(next(scholar_articles).get('bib', {}).get('author', ['Unknown'])),
#                         'year': next(scholar_articles).get('bib', {}).get('pub_year', 'N/A'),
#                         'url': f"https://scholar.google.com/scholar?q={'+'.join(next(scholar_articles).get('bib', {}).get('title', 'Untitled').split())}"
#                     }
#                     for _ in range(3)  # Limit to 3 articles
#                 ]
#             # YouTube video extraction via YouTube Data API
#             youtube_response = youtube_service.search().list(
#                     q=f"{topic} tutorial",
#                     part="snippet",
#                     type="video",
#                     maxResults=3
#                 ).execute()
#             step_resources['youtube_videos'] = [
#                     {
#                         'title': item['snippet']['title'],
#                         'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
#                     }
#                     for item in youtube_response['items']
#                 ]

#             # Fallback YouTube video extraction via SERPAPI
#             if not step_resources['youtube_videos']:
#                 try:
#                     serpapi_params = {
#                         "engine": "youtube",
#                         "search_query": f"{topic} tutorial",
#                         "api_key": SERPAPI_KEY
#                     }
#                     serpapi_response = requests.get(SERPAPI_BASE_URL, params=serpapi_params).json()
#                     step_resources['youtube_videos'] = [
#                         {
#                             'title': result['title'],
#                             'url': result['link']
#                         }
#                         for result in serpapi_response.get('video_results', [])[:3]
#                     ]
#                 except Exception as e:
#                     st.warning(f"Could not fetch YouTube videos via SERPAPI for {topic}: {e}")

#             resources['steps'].append(step_resources)

#         # Save learning path
#         st.session_state.user_data['learning_paths'][target] = {
#             'checklist': topics,
#             'completed': [],
#             'start_date': datetime.now().strftime("%Y-%m-%d"),
#             'last_updated': datetime.now().strftime("%Y-%m-%d"),
#             'learning_style': learning_style['style'],
#             'goal_type': goal_type,
#             'resources': resources
#         }

#         # Display Learning Path
#         st.success("Personalized Learning Path Generated!")
#         for step in resources['steps']:
#             with st.expander(f"📚 {step['topic']}"):
#                 st.write("### Recommended Resources")
#                 for resource in step['recommended_resources']:
#                     st.write(f"• {resource}")

#                 if step['scholarly_articles']:
#                     st.write("### 📚 Scholarly Articles")
#                     for article in step['scholarly_articles']:
#                         st.markdown(f"**[{article['title']}]({article['url']})**")
#                         st.write(f"Authors: {article['authors']}")
#                         st.write(f"Year: {article['year']}")
#                 else:
#                     st.write("No scholarly articles found for this step.")

#                 if step['youtube_videos']:
#                     st.write("### 🎥 YouTube Resources")
#                     for video in step['youtube_videos']:
#                         st.markdown(f"**[{video['title']}]({video['url']})**")
#                 else:
#                     st.write("No YouTube videos found for this step.")
#         return resources

#     except Exception as e:
#         st.error(f"Error generating learning path: {str(e)}")
#         return None

def generate_learning_path(target, learning_style, goal_type):
    # Simplified learning path generation
    path_prompt = f"""
    Create a comprehensive learning path for {target} optimized for {learning_style['style']} learners.
    Provide 6 - 10 key learning stages with specific, actionable topics.
    Format as a simple, clear list of key learning milestones.
    """
    try:
        # Generate core learning path
        path_response = model.generate_content(path_prompt)
        path_content = path_response.text.strip()

        # Split into topics
        topics = [item.strip() for item in path_content.split('\n') if item.strip() and item.strip().startswith(('1.', '2.', '3.', '4.' '•', '-'))]
        if not topics:
            topics = [item.strip() for item in path_content.split('\n') if item.strip()][:6]

        resources = {'steps': []}

        # Scholarly resources initialization
        for topic in topics:
            step_resources = {
                'topic': topic,
                'recommended_resources': [],
                'scholarly_articles': []
            }

            # Generate basic resources for each topic
            resource_prompt = f"""
            Suggest 3 key learning resources for the topic: {topic}
            Include:
            - 1 online course or tutorial
            - 1 book or reading resource
            - 1 practical project or exercise
            """
            try:
                resource_response = model.generate_content(resource_prompt)
                resources_text = resource_response.text.strip()

                # Parse resources
                step_resources['recommended_resources'] = [
                    resource.strip() for resource in resources_text.split('\n') 
                    if resource.strip() and not resource.strip().lower().startswith(('topic', 'suggestion'))
                ]
            except Exception as e:
                st.warning(f"Could not generate detailed resources for {topic}: {e}")

            # Scholarly articles extraction
            try:
                scholar_articles = scholarly.search_pubs(f"{target} {topic}")
                step_resources['scholarly_articles'] = [
                    {
                        'title': next(scholar_articles).get('bib', {}).get('title', 'Untitled'),
                        'authors': ', '.join(next(scholar_articles).get('bib', {}).get('author', ['Unknown'])),
                        'year': next(scholar_articles).get('bib', {}).get('pub_year', 'N/A'),
                        'url': f"https://scholar.google.com/scholar?q={'+'.join(next(scholar_articles).get('bib', {}).get('title', 'Untitled').split())}"
                    }
                    for _ in range(3)  # Limit to 3 articles
                ]
            except Exception as e:
                st.warning(f"Could not fetch scholarly articles for {topic}: {e}")

            resources['steps'].append(step_resources)

        # Save learning path
        st.session_state.user_data['learning_paths'][target] = {
            'checklist': topics,
            'completed': [],
            'start_date': datetime.now().strftime("%Y-%m-%d"),
            'last_updated': datetime.now().strftime("%Y-%m-%d"),
            'learning_style': learning_style['style'],
            'goal_type': goal_type,
            'resources': resources
        }

        # Display Learning Path
        st.success("Personalized Learning Path Generated!")
        for step in resources['steps']:
            with st.expander(f"📚 {step['topic']}"):
                st.write("### Recommended Resources")
                for resource in step['recommended_resources']:
                    st.write(f"• {resource}")

                if step['scholarly_articles']:
                    st.write("### 📚 Scholarly Articles")
                    for article in step['scholarly_articles']:
                        st.markdown(f"**[{article['title']}]({article['url']})**")
                        st.write(f"Authors: {article['authors']}")
                        st.write(f"Year: {article['year']}")
                else:
                    st.write("No scholarly articles found for this step.")
        return resources

    except Exception as e:
        st.error(f"Error generating learning path: {str(e)}")
        return None

# SERPAPI Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
from serpapi import GoogleSearch

def fetch_dynamic_courses(skill):
    """
    Fetch dynamic courses from curated lists or external APIs.
    """
    courses = []
    curated_courses = {
        "java full stack": [
            {
                "title": "Java Full Stack Developer Course",
                "url": "https://www.udemy.com/course/java-full-stack-developer/",
                "duration": "40 hours",
                "platform": "Udemy"
            },
            {
                "title": "Full Stack Development with React & Spring Boot",
                "url": "https://www.coursera.org/specializations/full-stack-react-spring",
                "duration": "20 weeks",
                "platform": "Coursera"
            }
        ],
        "data science": [
            {
                "title": "Data Science Specialization",
                "url": "https://www.coursera.org/specializations/jhu-data-science",
                "duration": "10 months",
                "platform": "Coursera"
            },
            {
                "title": "Data Science for Beginners",
                "url": "https://www.edx.org/course/data-science-for-beginners",
                "duration": "4 weeks",
                "platform": "edX"
            }
        ]
    }
    
    # Return curated courses for common skills
    skill_lower = skill.lower()
    if skill_lower in curated_courses:
        courses.extend(curated_courses[skill_lower])
    
    return courses


def fetch_communities(skill):
    """
    Fetch popular communities related to the skill.
    """
    communities = [
        {
            "name": "Java Enthusiasts Discord",
            "url": "https://discord.gg/java",
            "platform": "Discord"
        },
        {
            "name": "r/LearnJava",
            "url": "https://www.reddit.com/r/learnjava",
            "platform": "Reddit"
        },
        {
            "name": "Stack Overflow Java Tag",
            "url": "https://stackoverflow.com/questions/tagged/java",
            "platform": "Stack Overflow"
        }
    ]
    
    return communities

def about_us_tab():
    st.title("🎓 About Smart Learning Hub")
    
    # Main Description
    st.markdown("""
    ### Welcome to Smart Learning Hub
    Your AI-Powered Personal Learning Assistant that adapts to your unique learning style and goals.
    """)
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Active Users", value="1000+")
    with col2:
        st.metric(label="Courses Available", value="500+")
    with col3:
        st.metric(label="Success Rate", value="95%")
    
    # Key Features Section
    st.subheader("🌟 Key Features")
    
    # Using columns for feature layout
    feat_col1, feat_col2 = st.columns(2)
    
    with feat_col1:
        with st.expander("🎯 Personalized Learning Paths", expanded=True):
            st.write("""
            - Customized learning recommendations
            - AI-generated curriculum
            - Progress tracking with adaptive milestones
            - Dynamic course suggestions
            """)
            
        with st.expander("🧠 Advanced AI Integration", expanded=True):
            st.write("""
            - Gemini Pro Chat for intelligent tutoring
            - Smart document analysis with DocuMind
            - Media Mind for video content analysis
            - Real-time learning assistance
            """)
    
    with feat_col2:
        with st.expander("📚 Comprehensive Resource Hub", expanded=True):
            st.write("""
            - Curated courses from top platforms
            - Academic paper recommendations
            - Community resource integration
            - Interactive learning materials
            """)
            
        with st.expander("📊 Smart Progress Tracking", expanded=True):
            st.write("""
            - Detailed learning analytics
            - Achievement system with XP and levels
            - Study streak monitoring
            - Skill proficiency radar charts
            """)
    
    # Our Mission
    st.subheader("🎯 Our Mission")
    st.info("""
    Smart Learning Hub combines cutting-edge AI technology with proven educational methodologies 
    to create a personalized learning experience. We're committed to making quality education 
    accessible and adaptable to every learner's unique style and pace.
    """)
    
    # Learning Tools
    st.subheader("🛠️ Our Learning Tools")
    
    tool_col1, tool_col2, tool_col3 = st.columns(3)
    
    with tool_col1:
        st.markdown("### 🤖 Chatbot")
        st.write("AI-powered tutoring assistance available 24/7")
        
    with tool_col2:
        st.markdown("### 📄 DocuMind")
        st.write("Smart document analysis and learning material organization")
        
    with tool_col3:
        st.markdown("### 🎥 Media Mind")
        st.write("Intelligent video content analysis and summarization")
    
    # Benefits
    st.subheader("💡 Why Choose Smart Learning Hub?")
    benefits = {
        "Personalized Learning": "Adaptive paths based on your learning style",
        "AI-Powered": "Cutting-edge AI technology for enhanced learning",
        "Comprehensive Resources": "Access to diverse learning materials",
        "Progress Tracking": "Detailed analytics and achievement system",
        "Community Support": "Connect with fellow learners",
        "Flexible Learning": "Learn at your own pace, anywhere, anytime"
    }
    
    for title, description in benefits.items():
        with st.expander(title):
            st.write(description)
    
    # Contact Information
    st.subheader("📞 Contact Us")
    contact_col1, contact_col2 = st.columns(2)
    
    with contact_col1:
        st.markdown("#### Support")
        st.write("Email: support@smartlearninghub.com")
        st.write("Phone: +1 (555) 123-4567")
        
    with contact_col2:
        st.markdown("#### Social Media")
        st.write("Twitter: @SmartLearningHub")
        st.write("LinkedIn: Smart Learning Hub")
    
    
    st.subheader("📝 Resource Feedback")
    feedback = st.text_area("Help us improve! We value your opinion! Please take a moment to share your feedback on your recent experience with our site.")
    if st.button("Submit Feedback"):
        # Store feedback in user data
        if 'overall_feedback' not in st.session_state.user_data:
            st.session_state.user_data['overall_feedback'] = []
        st.session_state.user_data['overall_feedback'].append({
            'feedback': feedback,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        st.success("Thank you for your feedback!")
    # Version Information
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Smart Learning Hub - Your Personalized Learning Assistant</p>
            <p style='font-size: 0.8em'>Version 2.2</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Modified Learning Style Assessment Page
def show_learning_style_assessment():
    st.title("🧠 Discover Your Learning Style")
    
    learning_style_questions = {
        "scenario_1": {
            "question": "When learning something new, which method helps you remember best?",
            "options": [
                "Seeing diagrams and pictures",
                "Listening to explanations",
                "Hands-on practice",
                "Reading and taking notes"
            ]
        },
        "scenario_2": {
            "question": "When following instructions, you prefer to:",
            "options": [
                "Look at diagrams or demonstrations",
                "Hear verbal explanations",
                "Try it out yourself",
                "Read written instructions"
            ]
        },
        "scenario_3": {
            "question": "When studying, you tend to:",
            "options": [
                "Use charts and visual aids",
                "Discuss topics with others",
                "Create models or practice examples",
                "Make written summaries"
            ]
        }
    }
    
    with st.form("learning_style_form"):
        responses = {}
        for key, data in learning_style_questions.items():
            responses[key] = st.radio(
                data["question"],
                data["options"],
                key=f"style_{key}"
            )
        
        submit = st.form_submit_button("Analyze My Learning Style")
        
        if submit:
            with st.spinner("Analyzing your learning style..."):
                analysis = analyze_learning_style(responses)
                st.session_state.user_data['learning_style'] = analysis
                
                st.success(f"Your primary learning style: {analysis['style']}")
                
                tab1, tab2, tab3 = st.tabs(["Understanding", "Strategies", "Recommended Videos"])
                
                with tab1:
                    st.write("### Understanding Your Style")
                    st.write(analysis['explanation'])
                
                with tab2:
                    st.write("### Recommended Strategies")
                    for strategy in analysis['strategies']:
                        st.write(f"• {strategy}")
                    
                    st.write("### Recommended Tools")
                    for tool in analysis['tools']:
                        st.write(f"• {tool}")
                
                with tab3:
                    st.write("### Recommended Videos")
                    if analysis.get('recommended_videos'):
                        for video in analysis['recommended_videos']:
                            with st.expander(video['title']):
                                st.write(video['description'])
                                st.markdown(f"[Watch Video]({video['url']})")
                    else:
                        st.info("No specific video recommendations available.")


# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    show_login_page()
else:
    # Learning Style Assessment Function
    def analyze_learning_style(responses):
        prompt = f"""
        Based on the following learning style assessment responses:
        {json.dumps(responses, indent=2)}
        
        Please analyze the learning style and provide a response in the following format:
        {{
            "style": "Visual/Auditory/Kinesthetic/Reading-Writing",
            "explanation": "Detailed explanation of the learning style",
            "strategies": ["Strategy 1", "Strategy 2", "Strategy 3"],
            "tools": ["Tool/Resource 1", "Tool/Resource 2", "Tool/Resource 3"],
            "recommended_videos": [
            {{
                "title": "Video Title 1",
                "url": "https://www.youtube.com/watch?v=...",
                "description": "Brief description of the video content"
            }},
            {{
                "title": "Video Title 2",
                "url": "https://www.youtube.com/watch?v=...",
                "description": "Brief description of the video content"
            }}
        }}
        Ensure the video recommendations are real, popular educational videos relevant to the learning style.
        """
        
        try:
            response = model.generate_content(prompt)
            # Ensure the response is properly formatted as JSON
            result = response.text.strip()
            if not result.startswith('{'):
                result = result[result.find('{'):result.rfind('}')+1]
            return json.loads(result)
        except Exception as e:
            st.error(f"Error analyzing learning style. Please try again. Error: {str(e)}")
            return {
                "style": "General",
                "explanation": "Unable to determine specific learning style. Using general learning approach.",
                "strategies": ["Mixed learning approach", "Multiple resource types", "Regular practice"],
                "tools": ["Online courses", "Educational videos", "Practice exercises"],
                "recommended_videos": []
            }

    # Resource Generation Function
    def generate_resources(path, learning_style="General"):
        prompt = f"""
        As an expert educator, suggest detailed, high-quality learning resources for {path} tailored for {learning_style} learners.
        Focus only on widely available, reputable sources.
        
        Structure your response exactly like this:

        ONLINE COURSES:
        - Provide 3-4 popular courses from Coursera, edX, Udemy with:
            - Exact course names that exist on these platforms
            - Brief description focusing on key benefits
            - Approximate duration and difficulty level
            - Main topics covered
        
        BOOKS AND MATERIALS:
        - List 3-4 widely recognized books with:
            - Full title and author name
            - Publisher and year if recent
            - Key topics and target audience
            - Why it's particularly good for {learning_style} learners
        
        PRACTICE PLATFORMS:
        - Suggest 2-3 interactive platforms with:
            - Platform name and main focus
            - Types of exercises/projects available
            - Skill level requirements
            - Any free vs paid considerations
        
        LEARNING COMMUNITIES:
        - Recommend 2-3 active communities with:
            - Community name and platform (Reddit, Discord, Stack Exchange, etc.)
            - Main focus and activity type
            - Why it's valuable for learners
            - Any entry requirements or guidelines
        
        VIDEO RESOURCES:
        - Suggest 2-3 high-quality video channels or series with:
            - Channel/Creator name
            - Content style and format
            - Best playlists or series to start with
            - Why it works well for {learning_style} learning style

        Focus on currently active and maintained resources. For each resource, explain why it's particularly suitable for {learning_style} learners.
        Do not include specific URLs, as these can change. Instead, provide enough information for users to easily find the resources.
        """
        
        try:
            response = model.generate_content(prompt)
            resources_content = response.text.strip()
            return resources_content
        except Exception as e:
            return f"Error generating resources: {str(e)}"

    def show_resources_page():
        st.title("📚 Learning Resources")
        
        if not st.session_state.user_data['learning_paths']:
            st.info("Please create a learning path first to get personalized resources!")
            return

        # Get user's learning style
        learning_style = st.session_state.user_data.get('learning_style', {}).get('style', 'General')
        
        # Path selection
        selected_path = st.selectbox(
            "Select Learning Path:",
            list(st.session_state.user_data['learning_paths'].keys())
        )
        
        if selected_path:
            with st.spinner("Generating personalized resources..."):
                resources_content = generate_resources(selected_path, learning_style)
                
                # Display resources by category
                categories = [
                    "ONLINE COURSES",
                    "BOOKS AND MATERIALS",
                    "PRACTICE PLATFORMS",
                    "LEARNING COMMUNITIES",
                    "VIDEO RESOURCES"
                ]
                
                for category in categories:
                    with st.expander(f"📌 {category}", expanded=True):
                        try:
                            # Extract category content
                            if category in resources_content:
                                section = resources_content.split(category + ":")[1]
                                # Get content until next category or end
                                end_pos = len(section)
                                for next_cat in categories:
                                    if next_cat in section:
                                        end_pos = min(end_pos, section.index(next_cat))
                                content = section[:end_pos].strip()
                                
                                # Display content with improved formatting
                                for line in content.split('\n'):
                                    if line.strip():
                                        if line.startswith('-'):
                                            st.markdown(f"**{line.strip('-').strip()}**")
                                        else:
                                            st.write(line.strip())
                        except Exception as e:
                            st.write("No resources available for this category.")
            # Add refresh button
            if st.button("🔄 Refresh Resources"):
                                       st.write("No resources available for this category.")
        
        
        # Add feedback section
        st.markdown("---")
        st.subheader("📝 Resource Feedback")
        feedback = st.text_area("Help us improve! Let us know if any resources are outdated or particularly helpful:")
        if st.button("Submit Feedback"):
            # Store feedback in user data
            if 'resource_feedback' not in st.session_state.user_data:
                st.session_state.user_data['resource_feedback'] = []
            st.session_state.user_data['resource_feedback'].append({
                'path': selected_path,
                'feedback': feedback,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("Thank you for your feedback!")
    
    # Dashboard Components
    def create_study_streak_chart():
        dates = pd.date_range(end=datetime.now(), periods=30)
        activity = [random.choice([0, 1, 1, 1]) for _ in range(30)]  # Simulate activity
        df = pd.DataFrame({'date': dates, 'activity': activity})
        
        fig = px.bar(df, x='date', y='activity', 
                    title='Study Streak - Last 30 Days',
                    labels={'activity': 'Study Sessions', 'date': 'Date'})
        return fig

    def calculate_learning_metrics():
        total_paths = len(st.session_state.user_data['learning_paths'])
        completed_items = sum(len(path['completed']) for path in st.session_state.user_data['learning_paths'].values())
        total_items = sum(len(path['checklist']) for path in st.session_state.user_data['learning_paths'].values())
        completion_rate = (completed_items / total_items * 100) if total_items > 0 else 0
        
        # Calculate active days
        # Create a set of unique dates when learning path items were completed
        active_dates = set()
        for path in st.session_state.user_data['learning_paths'].values():
            # Assuming completed items have timestamps or start dates
            start_date = datetime.strptime(path.get('start_date', datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d")
            active_dates.add(start_date.date())
        
        # Estimate study hours based on learning path progress
        # Assume each completed item represents approximately 1 hour of study
        study_hours = completed_items

        return {
            'total_paths': total_paths,
            'completion_rate': completion_rate,
            'active_days': len(active_dates),  # Simulated data
            'study_hours': study_hours  # Simulated data
        }
    
    def show_enhanced_dashboard():
        st.title("🚀 Advanced Learning Dashboard")
        
        # Gamification Section
        st.header("🏆 Learning Profile")
        
        # Calculate Learning XP and Level
        def calculate_learning_xp():
            xp = 0
            for path in st.session_state.user_data['learning_paths'].values():
                # More XP for completed items
                xp += len(path['completed']) * 50
                # Bonus XP for entire path completion
                if len(path['completed']) == len(path['checklist']):
                    xp += 500
            return xp
        
        xp = calculate_learning_xp()
        level = min(int(xp ** 0.5 / 10), 10)  # Logarithmic level progression
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Learning Level", level)
        with col2:
            st.metric("Total XP", xp)
        with col3:
            st.progress(min(xp % 1000 / 1000, 1), text="Next Level Progress")
        
        # Skill Radar Chart
        st.header("🌟 Skill Proficiency")
        skills = {
            "Technical Skills": random.uniform(0.3, 0.9),
            "Soft Skills": random.uniform(0.3, 0.9),
            "Communication": random.uniform(0.3, 0.9),
            "Problem Solving": random.uniform(0.3, 0.9),
            "Creativity": random.uniform(0.3, 0.9)
        }
        
        # Plotly Radar Chart
        fig = go.Figure(data=go.Scatterpolar(
            r=list(skills.values()),
            theta=list(skills.keys()),
            fill='toself'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)
        
        # Learning Challenges
        st.header("🎯 Weekly Learning Challenges")
        challenges = [
            "Complete 3 learning modules",
            "Spend 5 hours studying",
            "Try a new learning resource",
            "Engage in a peer study session"
        ]
        
        for challenge in challenges:
            st.checkbox(challenge)
        
        # Achievements Section
        st.header("🏅 Recent Achievements")
        achievements = [
            "🔥 10-Day Learning Streak",
            "📚 Completed Python Basics Path",
            "🧠 Mastered Machine Learning Fundamentals"
        ]
        
        for achievement in achievements:
            st.markdown(f"- {achievement}")
        
        # Recommended Next Steps
        st.header("🚀 Recommended Next Steps")
        recommendations = [
            "Continue your Machine Learning path",
            "Explore advanced Python concepts",
            "Join a coding community"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")

    # Replace the existing dashboard function
    def show_dashboard():
        show_enhanced_dashboard()

    def get_learning_recommendations():
        if not st.session_state.user_data.get('learning_style'):
            return ["Complete your learning style assessment to get personalized recommendations!"]
        
        style = st.session_state.user_data['learning_style']['style']
        return [
            f"Based on your {style} learning style:",
            "• " + random.choice([
                "Try creating mind maps for complex topics",
                "Record yourself explaining concepts",
                "Use interactive coding environments",
                "Write detailed study notes"
            ]),
            "• " + random.choice([
                "Take regular breaks every 25 minutes",
                "Join study groups for discussion",
                "Practice with real-world projects",
                "Create flashcards for key concepts"
            ]),
            "• " + random.choice([
                "Watch video tutorials with closed captions",
                "Explain concepts to others",
                "Use physical objects to model problems",
                "Summarize readings in your own words"
            ])
        ]
    # Main Dashboard Page
    def show_dashboard():
        st.title("🎓 Learning Dashboard")
        
        def calculate_learning_xp():
            xp = 0
            for path in st.session_state.user_data['learning_paths'].values():
                # More XP for completed items
                xp += len(path['completed']) * 50
                # Bonus XP for entire path completion
                if len(path['completed']) == len(path['checklist']):
                    xp += 500
            return xp
        
        xp = calculate_learning_xp()
        level = min(int(xp ** 0.5 / 10), 10)
        
        # Top metrics
        metrics = calculate_learning_metrics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Learning Paths", metrics['total_paths'])
        with col2:
            st.metric("Completion Rate", f"{metrics['completion_rate']:.1f}%")
        with col3:
            st.metric("Active Days", metrics['active_days'])
        with col4:
            st.metric("Total XP", xp)
        
        
        # Study streak chart
        # st.plotly_chart(create_study_streak_chart(), use_container_width=True)
        
        # Recent activity and recommendations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Recent Activity")
            if st.session_state.user_data['learning_paths']:
                for path, data in st.session_state.user_data['learning_paths'].items():
                    with st.expander(f"📚 {path}"):
                        progress = len(data['completed']) / len(data['checklist']) * 100
                        st.progress(progress / 100)
                        st.write(f"Progress: {progress:.1f}%")
                        st.write("Recent completions:")
                        for item in data['completed'][-3:]:
                            st.write(f"✅ {item}")
            else:
                st.info("Start a learning path to track your progress!")
        
        with col2:
            st.subheader("💡 Recommendations")
            recommendations = get_learning_recommendations()
            for rec in recommendations:
                st.write(rec)
            
            if st.button("Get New Tips"):
                st.rerun()

    # Sidebar Navigation
    with st.sidebar:
        st.image("https://media.licdn.com/dms/image/v2/D4D12AQHGJ3I-fJELnA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1715845287543?e=2147483647&v=beta&t=D6L7ZHUy8qh3aFQsyeAGv0wgXB5UskmGNIwe2OETaig", caption="Smart Learning Hub")
        st.write(f"Welcome, {st.session_state['username']}!")
        if st.button("Logout"):
            # Save user data before logout
            save_user_data(st.session_state['username'], st.session_state.user_data)
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state.user_data = None
            st.rerun()
            
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "Learning Style Assessment", "🔰 Learning Path", "📈 Progress Tracking", "📚 Resources", "👩🏻‍🏫 Courses", "🗂️ DocuMind", "🎥 Media Mind","🤖 Chatbot", "🛈 About Us"]
        )

# Resources Page with Error Handling
    if page == "🤖 Chatbot":
        gemini_pro_chat_tab()
    elif page == "📚 Resources":
        show_resources_page()
    elif page == "🎥 Media Mind":
        video_analyzer_tab()
    elif  page == "🗂️ DocuMind":
        document_retrieval_tab()
    elif page == "🛈 About Us":
        about_us_tab()
    elif page== "👩🏻‍🏫 Courses":
        courses_tab()
        # Dynamic section based on user-selected skill
        st.markdown("## 🔍 Recommended Courses and Communities")
        if "learning_paths" in st.session_state.user_data and st.session_state.user_data["learning_paths"]:
            learning_paths = st.session_state.user_data["learning_paths"]
            selected_skill = st.selectbox("Select a Learning Path:", list(learning_paths.keys()))
            
            if selected_skill:
                st.write(f"### Recommendations for {selected_skill}")

                # Dynamic courses
                with st.spinner("Fetching courses..."):
                    dynamic_courses = fetch_dynamic_courses(selected_skill)
                    if dynamic_courses:
                        st.subheader("🆕 Recommended Courses")
                        for course in dynamic_courses:
                            st.markdown(f"**[{course['title']}]({course['url']})**")
                            st.write(f"Platform: {course['platform']}, Duration: {course['duration']}")
                    else:
                        st.info("No dynamic courses found for this skill.")

                # Communities
                with st.spinner("Fetching communities..."):
                    communities = fetch_communities(selected_skill)
                    if communities:
                        st.subheader("💬 Popular Communities")
                        for community in communities:
                            st.markdown(f"**[{community['name']}]({community['url']})**")
                            st.write(f"Platform: {community['platform']}")
                    else:
                        st.info("No communities found for this skill.")

                # Bookmarking feature
                st.markdown("### 📌 Bookmark Your Favorites")
                if "bookmarks" not in st.session_state:
                    st.session_state.bookmarks = []

                for course in dynamic_courses:
                    if st.button(f"Bookmark {course['title']}"):
                        st.session_state.bookmarks.append(course)
                        st.success(f"Bookmarked {course['title']}!")

                st.markdown("### ⭐ Your Bookmarks")
                if st.session_state.bookmarks:
                    for bookmark in st.session_state.bookmarks:
                        st.markdown(f"**[{bookmark['title']}]({bookmark['url']})**")
                        st.write(f"Platform: {bookmark['platform']}, Duration: {bookmark['duration']}")
                else:
                    st.info("No bookmarks yet.")
        else:
            st.warning("No learning paths found. Please create a learning path to get recommendations.")

    # Learning Style Assessment Page
    elif page == "Learning Style Assessment":
        st.title("🧠 Discover Your Learning Style")
        
        learning_style_questions = {
            "scenario_1": {
                "question": "When learning something new, which method helps you remember best?",
                "options": [
                    "Seeing diagrams and pictures",
                    "Listening to explanations",
                    "Hands-on practice",
                    "Reading and taking notes"
                ]
            },
            "scenario_2": {
                "question": "When following instructions, you prefer to:",
                "options": [
                    "Look at diagrams or demonstrations",
                    "Hear verbal explanations",
                    "Try it out yourself",
                    "Read written instructions"
                ]
            },
            "scenario_3": {
                "question": "When studying, you tend to:",
                "options": [
                    "Use charts and visual aids",
                    "Discuss topics with others",
                    "Create models or practice examples",
                    "Make written summaries"
                ]
            }
        }
        
        with st.form("learning_style_form"):
            responses = {}
            for key, data in learning_style_questions.items():
                responses[key] = st.radio(
                    data["question"],
                    data["options"],
                    key=f"style_{key}"
                )
            
            submit = st.form_submit_button("Analyze My Learning Style")
            
            if submit:
                with st.spinner("Analyzing your learning style..."):
                    analysis = analyze_learning_style(responses)
                    st.session_state.user_data['learning_style'] = analysis
                    
                    st.success(f"Your primary learning style: {analysis['style']}")
                    st.write("### Understanding Your Style")
                    st.write(analysis['explanation'])
                    
                    st.write("### Recommended Strategies")
                    for strategy in analysis['strategies']:
                        st.write(f"• {strategy}")
                    
                    st.write("### Recommended Tools")
                    for tool in analysis['tools']:
                        st.write(f"• {tool}")

    # Learning Path Page
    elif page == "🔰 Learning Path":
        st.title("🎯 Create Your Learning Path")
    
        learning_style = st.session_state.user_data.get('learning_style')
        if not learning_style:
            st.warning("Complete the Learning Style Assessment first for a personalized learning path!")
        else:
            st.success(f"Creating path optimized for your {learning_style['style']} learning style")
            
            goal_type = st.radio("Select your goal type:", 
                                ["New Career Path", "Skill Enhancement", "Certification"])
            
            target = st.text_input("Enter your target role or skill:")
            
            if target and st.button("Generate Learning Path"):
                with st.spinner("Generating your personalized learning path..."):
                    # Use the new function with learning style
                    generate_learning_path(target, learning_style, goal_type)

        
    # Progress Tracking Page
    elif page == "📈 Progress Tracking":
        st.title("📈 Progress Tracking")
        
        for path, data in st.session_state.user_data['learning_paths'].items():
            st.subheader(f"Path: {path}")
            
            # Progress calculation
            total_items = len(data['checklist'])
            completed_items = len(data['completed'])
            progress = (completed_items / total_items * 100) if total_items > 0 else 0
            
            # Display progress
            st.progress(progress / 100)
            st.write(f"Progress: {progress:.1f}%")
            
            # Checklist with detailed dropdowns
            for i, item in enumerate(data['resources']['steps']):
                topic = item['topic']
                details = item.get('recommended_resources', [])
                scholarly_articles = item.get('scholarly_articles', [])
                youtube_videos = item.get('youtube_videos', [])

                # Checkbox for topic completion
                checked = topic in data['completed']
                if st.checkbox(topic, value=checked, key=f"check_{path}_{i}"):
                    if topic not in data['completed']:
                        data['completed'].append(topic)
                elif topic in data['completed']:
                    data['completed'].remove(topic)

                # Expandable details
                with st.expander(f"View Details for: {topic}"):
                    st.write("### Recommended Resources")
                    for resource in details:
                        st.write(f"• {resource}")

                    st.write("### Scholarly Articles")
                    for article in scholarly_articles:
                        st.markdown(f"**[{article['title']}]({article['url']})**")
                        st.write(f"Authors: {article['authors']}")
                        st.write(f"Year: {article['year']}")

                    st.write("### YouTube Resources")
                    for video in youtube_videos:
                        st.markdown(f"**[{video['title']}]({video['url']})**")
            
            # Save the updated progress
            data['last_updated'] = datetime.now().strftime("%Y-%m-%d")
    
    elif page == "📊 Dashboard":
        show_dashboard()

    # Auto-save user data periodically
    if st.session_state.get('username'):
        save_user_data(st.session_state['username'], st.session_state.user_data)
    load_dotenv()


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Smart Learning Hub - Your Personalized Learning Assistant</p>
        <p style='font-size: 0.8em'>Version 2.2</p>
    </div>
    """,
    unsafe_allow_html=True
)
