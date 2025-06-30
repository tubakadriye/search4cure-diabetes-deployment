#streamlit run app.py
# === Agent Executor Setup ===
from agent.agent_executer import build_agent_executor
from graph.agent_graph import build_agent_graph, agent_node

from embeddings.gemini_text_embedding import get_gemini_embedding
import streamlit as st
from loaders.user_pdf_loader import load_user_pdfs_from_folder, load_user_pdf_from_url
from loaders.arxiv_loader import ArxivPDFLoader
from loaders.pubmed_loader import PubMedPDFLoader
import os
from utils.csv_processing import process_and_upload_csv
from embeddings.clip import get_clip_embedding
from db.mongodb_client import mongodb_client
from db.index_utils import create_vector_index, create_multivector_index
from multimodal.pdf_processing import process_and_embed_docs
import base64
from agent_tools.article_tools import vector_search_image_tool
DB_NAME = "diabetes_data"
response_collection = mongodb_client[DB_NAME]["responses"]

# === Agent Execution Setup ===
agent_executor = build_agent_executor()

def chatbot_node(state):
    return agent_node(state, agent_executor.agent, name="chatbot")

def tool_node(state):
    return agent_node(state, agent_executor.agent, name="tools")

graph = build_agent_graph(chatbot_node, tool_node)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Search4Cure.AI: Diabetes", layout="wide")
st.title("🔬 Search4Cure.AI: Diabetes")
st.markdown("**Search4Cure.AI: Diabetes** is a multimodal research assistant designed to help you explore, analyze, and embed diabetes-related scientific documents, images, and datasets using AI-powered search and visualization.")


# Setup MongoDB
db = mongodb_client["diabetes_data"]
pdf_collection = db["docs_multimodal"]


# --- Session States ---
for key in ["user_pdfs", "embedded_docs", "search_results"]:
    st.session_state.setdefault(key, [])

# --- Human-in-the-Loop HITL Session Keys ---
for key in ["agent_raw_response", "agent_approved_text", "agent_review_mode"]:
    st.session_state.setdefault(key, "" if "response" in key else False)

# --- Sidebar Upload ---
with st.sidebar:
    st.header("Upload PDFs, CSVs, or Images")

    # PDFs from file
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        user_upload_dir = "user_uploads"
        os.makedirs(user_upload_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            path = os.path.join(user_upload_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
        st.success(f"Saved {len(uploaded_files)} PDF(s).")
        #user_pdfs = load_user_pdfs_from_folder(user_upload_dir)
        # Load PDFs from folder and append to session state
        st.session_state.user_pdfs += load_user_pdfs_from_folder(user_upload_dir)

    # PDFs from URL
    pdf_url = st.text_input("Enter PDF URL:")
    if pdf_url:
        #user_pdfs = load_user_pdf_from_url(pdf_url)
        # Append loaded PDF(s) from URL to session state list
        st.session_state.user_pdfs += load_user_pdf_from_url(pdf_url)

    # --- CSV Upload ---
    uploaded_csv = st.file_uploader("📊 Upload CSV", type=["csv", "xlsx", "xls", "json"])
    csv_name = uploaded_csv.name.rsplit('.', 1)[0] if uploaded_csv else ""
    st.text_input("CSV Name:", value=csv_name)

    # --- Optional Arxiv & PubMed fetching ---
    add_research = st.checkbox("🔍 Redo Diabetes research from Arxiv & PubMed")

    #all_pdfs = user_pdfs

    if add_research:
        if st.button("📥 Load Arxiv & PubMed PDFs"):
            with st.spinner("Loading Arxiv and PubMed PDFs..."):
                arxiv_pdfs = ArxivPDFLoader(query="Diabetes").download_pdfs()
                pubmed_pdfs = PubMedPDFLoader(query="Diabetes").download_pdfs()
                #all_pdfs += arxiv_pdfs + pubmed_pdfs
                st.session_state.user_pdfs += arxiv_pdfs + pubmed_pdfs
                st.success(f"Loaded {len(arxiv_pdfs) + len(pubmed_pdfs)} research PDFs.")

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    default_name = uploaded_image.name.rsplit('.', 1)[0] if uploaded_image else ""
    custom_image_name = st.text_input("Image name:", default_name)

    # -- Process all button ---
    if st.button("📥 Process All"):
        if not st.session_state.user_pdfs and not uploaded_image:
            st.warning("Please upload PDFs or an image first.")
        else:
            with st.spinner("Embedding and saving..."):
                embedded_docs = process_and_embed_docs(
                    pdfs=st.session_state.user_pdfs,
                    uploaded_image=uploaded_image,
                    image_name=custom_image_name,
                    get_embedding=get_clip_embedding ## for image embedding
                )
                st.session_state.embedded_docs = embedded_docs
                if embedded_docs:
                    # Now embed documents
                    #embedded_docs = embed_docs_with_clip(docs, get_clip_embedding)                   
                    # Insert into MongoDB
                    pdf_collection.insert_many(embedded_docs)                 

                if uploaded_csv:
                    process_and_upload_csv(
                        uploaded_csv,
                        datasets_col=db["datasets"],
                        data_col=db["records_embeddings"],
                        embedding_fn=get_gemini_embedding
                    )


                st.success(f"✅ Processed, uploaded, and embedded {len(embedded_docs)} items.")
                # Example: show first 3 embedded docs metadata or images
                for doc in embedded_docs[:3]:
                    st.image(doc["image"], caption=f'{doc["pdf_title"]} - Page {doc["page_number"]}', width=150)


# --- Search Interface  ---

st.markdown("<h2 style='text-align:center'>🔍 Search Query</h2>", unsafe_allow_html=True)
query = st.text_input("", placeholder="Enter your search query here...", key="search_query", max_chars=200)

search_button = st.button("Search")

# Add image upload in query section
uploaded_query_image = st.file_uploader("Or upload an image to query about:", type=["png", "jpg", "jpeg"], key="query_image")

image_base64 = None
# Show the uploaded image (if provided)
if uploaded_query_image:
    st.image(uploaded_query_image, caption="🔍 Uploaded Query Image", use_container_width=True)      
search_button = st.button("Search")

if search_button:
    if not query.strip() and not uploaded_query_image:
        st.warning("Please enter a query or upload an image.")
    else:
        with st.spinner("Using Diabetes Research Asisstant Agent to answer..."):
            try:
                # Case 1: Both image and text query provided
                if uploaded_query_image and query.strip():
                    image_bytes = uploaded_query_image.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                    # If your agent_executor can handle multimodal input, pass both
                    combined_input = {
                        "input": f"query: {query} , image_base64: {image_base64}"
                    }
                    agent_response = agent_executor.invoke(combined_input)
                    response_text = agent_response.get("output", str(agent_response))

                    st.session_state.agent_raw_response = response_text
                    st.session_state.agent_review_mode = True
                    st.markdown("### 🖼️ + 📄 Multimodal Search Results")
                    st.markdown(response_text)
                # Case 2: Only image uploaded
                elif uploaded_query_image:
                    image_bytes = uploaded_query_image.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    image_search_result = vector_search_image_tool.invoke({
                        "image_base64": image_base64
                    })
                    st.markdown("### 🖼️ Image-Based Search Results")
                    st.markdown(image_search_result)

                # Run text-based query using the agent
                elif query.strip():                
                    agent_response = agent_executor.invoke({"input": query})
                    response_text = agent_response.get("output", str(agent_response))                    
                    st.markdown("### 📄 Text-Based Search Results")
                    st.markdown(response_text)
            except Exception as e:
                st.error(f"Error during text-based search: {e}")
            st.session_state.agent_raw_response = response_text
            st.session_state.agent_review_mode = True  # activate HITL mode



# --- Human-in-the-Loop Review Interface ---
if st.session_state.agent_review_mode:
    st.markdown("### 🤖 Suggested Answer by Agent")
             
    st.session_state.agent_approved_text = st.text_area(
        "Edit or approve the agent's response:",
        value=st.session_state.agent_raw_response,
        height=200,
        key="editable_response"
    )


    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve"):
            st.session_state.agent_review_mode = False
            st.success("✅ Approved Response:")
            st.markdown(f"**{st.session_state.agent_approved_text}**")
            # Here you can save the approved response to a database/log
            response_collection.insert_one({"query": query, "response": st.session_state.agent_approved_text})
    

    with col2:
        if st.button("🔄 Regenerate Agent Response"):
            with st.spinner("Regenerating answer..."):
                new_agent_response = agent_executor.invoke({"input": query})
                new_response_text = new_agent_response.get("output", str(new_agent_response))
                st.session_state.agent_approved_text = new_response_text
                st.experimental_rerun()

    # Expert input section
    st.markdown("### 📝 Expert Input")
    expert_input = st.text_area(
        "Expert can edit or write their own answer below:",
        value=st.session_state.agent_approved_text,
        height=200,     
        key="expert_response"
        )

    if st.button("✅ Submit Expert Answer"):
        st.session_state.agent_approved_text = expert_input
        st.session_state.agent_review_mode = False
        st.success("✅ Expert-approved Response:")
        st.markdown(f"**{st.session_state.agent_approved_text}**")
        response_collection.insert_one({"query": query, "expert_input": st.session_state.agent_approved_text})



            


            






            

