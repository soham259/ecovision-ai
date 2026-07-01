"""
app.py
------
EcoVision AI - Smart Waste Classification and Recycling Assistant.

Main Streamlit application entry point. This file contains ONLY UI/page
code. All model loading, inference, and business logic is imported from
utils.py and reused as-is (no helper functions are redefined here).

Pages:
    - Home
    - Image Prediction
    - AI Assistant
    - About
"""

import os
import tempfile

import streamlit as st

from utils import (
    load_cnn_model,
    load_intent_model,
    load_knowledge_base,
    load_gemini_client,
    generate_response,
    display_prediction,
)

# ==========================================================================================
# PAGE CONFIGURATION  (must be the first Streamlit call)
# ==========================================================================================
st.set_page_config(
    page_title="EcoVision AI | Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================================
# CUSTOM CSS
# ==========================================================================================
def load_custom_css():
    """Injects the project's custom stylesheet (assets/css/style.css) if present."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def apply_contrast_fixes():
    """
    Forces readable text color inside light/white cards in the sidebar
    (Session Stats metrics + System Status expander). BUGFIX: the project
    stylesheet renders these as white cards but leaves the text white too,
    making the numbers and labels invisible. This runs AFTER the custom
    stylesheet so !important overrides win regardless of load order.
    """
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="stMetric"] {
            background-color: #ffffff !important;
            border-radius: 10px;
            padding: 8px 4px;
        }
        section[data-testid="stSidebar"] [data-testid="stMetricValue"],
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
        section[data-testid="stSidebar"] [data-testid="stMetricDelta"] {
            color: #0f3d3e !important;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
        section[data-testid="stSidebar"] [data-testid="stExpander"] p,
        section[data-testid="stSidebar"] [data-testid="stExpander"] li {
            color: #0f3d3e !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_custom_css()
apply_contrast_fixes()


# ==========================================================================================
# SESSION STATE INITIALIZATION
# ==========================================================================================
def init_session_state():
    defaults = {
        "assistant_history": [],       # AI Assistant chat log
        "assistant_image_path": None,  # Persisted image for AI Assistant page
        "assistant_image_name": None,
        "total_predictions": 0,
        "total_recyclable": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==========================================================================================
# SHARED HELPERS (UI-only, not business logic)
# ==========================================================================================
def save_uploaded_file(uploaded_file, suffix=None):
    """Persists an in-memory uploaded file to a temp path so utils.py functions
    (which expect file paths) can consume it.

    BUGFIX: suffix is now optional. If not provided, the real extension of the
    uploaded file is used when available (e.g. .jpg/.png from st.file_uploader).
    """
    if suffix is None:
        name = getattr(uploaded_file, "name", None)
        suffix = os.path.splitext(name)[1] if name else ".jpg"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def cleanup_temp_file(path):
    """Best-effort removal of a temp file created by save_uploaded_file().
    OPTIMIZATION: prevents unbounded temp file accumulation on disk across
    a long-running session (NamedTemporaryFile(delete=False) never cleans
    up on its own)."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass  # File may be locked/in use - safe to ignore


def render_result_card(result):
    """Renders a generate_response() result using cards, columns, metrics,
    and expanders. Uses display_prediction() from utils.py for text content."""
    if result.get("Status") == "Error":
        st.error(result.get("Message", "Something went wrong. Please try again."))
        return

    st.success("✅ Prediction completed successfully!")

    st.session_state["total_predictions"] += 1
    if result["Recyclable"]:
        st.session_state["total_recyclable"] += 1

    with st.container(border=True):
        st.markdown(f"### 🗂️ Detected Item: **{result['Detected Item'].title()}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{result['Confidence']}%")
        m2.metric("Recyclable", "Yes ✅" if result["Recyclable"] else "No ❌")
        m3.metric("Recommended Bin", result["Recommended Bin"])

        st.markdown(f"**Intent detected:** `{result['Intent']}`")
        st.info(result["Response"])

        tab1, tab2, tab3 = st.tabs(
            ["🧪 Material & Environment", "💡 Tips", "📊 Extra Info"]
        )

        with tab1:
            st.markdown(f"**Material:** {result['Material']}")
            st.markdown(f"**Environmental Impact:** {result['Environmental Impact']}")
            st.markdown(f"**Decomposition:** {result['Decomposition']}")
            st.markdown(f"**Carbon Impact:** {result['Carbon Impact']}")

        with tab2:
            for tip in result["Tips"]:
                st.markdown(f"- {tip}")

        with tab3:
            st.markdown(f"**Common Uses:** {', '.join(result['Common Uses'])}")
            st.markdown(f"**Environmental Score:** {result['Environmental Score']} / 10")
            st.progress(min(result["Environmental Score"] / 10, 1.0))
            st.caption(f"💬 {result['Interesting Fact']}")

        with st.expander("📄 View Full Formatted Report"):
            st.markdown(display_prediction(result))


# ==========================================================================================
# SIDEBAR
# ==========================================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h1>♻️ EcoVision AI</h1>
                <p>Smart Waste Classification & Recycling Assistant</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📷 Image Prediction", "🤖 AI Assistant", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.divider()

        with st.expander("⚙️ System Status", expanded=False):
            st.caption("Models load automatically on first use and are cached.")
            st.markdown("- MobileNetV2: Image Classification")
            st.markdown("- DistilBERT: Intent Detection")
            st.markdown("- Knowledge Base: Recycling Data")
            gemini_status = "🟢 Connected" if load_gemini_client() is not None else "⚪ Not configured"
            st.markdown(f"- Gemini AI Answers: {gemini_status}")

        st.divider()
        st.caption("Built with Streamlit • TensorFlow • Transformers")

        return page


# ==========================================================================================
# PAGE: HOME
# ==========================================================================================
def page_home():
    st.markdown(
        """
        <div class="hero-card">
            <h1>♻️ EcoVision AI</h1>
            <p>Smart Waste Classification and Recycling Assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Platform Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Waste Categories", "10")
    col2.metric("AI Models", "2")
    col3.metric("Input Modes", "Image + Text")
    col4.metric("Knowledge Fields", "13+")

    st.markdown("### 🚀 Core Capabilities")
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("#### 📷 Image Classification")
            st.write(
                "Upload or capture a photo of a waste item. MobileNetV2 identifies "
                "its category with confidence scoring."
            )

    with c2:
        with st.container(border=True):
            st.markdown("#### 🧠 Smart Recommendations")
            st.write(
                "A curated knowledge base returns disposal guidance, recyclability, "
                "environmental impact, and tips."
            )

    st.markdown("### 🧭 How It Works")
    with st.expander("View the end-to-end pipeline", expanded=True):
        t1, t2, t3, t4 = st.tabs(["1️⃣ Input", "2️⃣ Recognition", "3️⃣ Matching", "4️⃣ Response"])
        with t1:
            st.write("Provide a waste image (upload/camera) and a typed question.")
        with t2:
            st.write("MobileNetV2 classifies the image; DistilBERT detects the question's intent.")
        with t3:
            st.write("The predicted class and intent are matched against the knowledge base.")
        with t4:
            st.write("A structured, actionable recycling recommendation is generated and displayed.")

    st.info("👈 Use the sidebar to navigate to **Image Prediction** or the **AI Assistant**.")


# ==========================================================================================
# PAGE: IMAGE PREDICTION
# ==========================================================================================
def page_image_prediction():
    st.markdown("## 📷 Image-Based Waste Prediction")
    st.caption("Upload a photo and ask a question to get an instant recycling recommendation.")

    col_input, col_result = st.columns([1, 1.4], gap="large")

    with col_input:
        with st.container(border=True):
            st.markdown("#### Upload Details")

            img_tab_upload, img_tab_camera = st.tabs(["📁 Upload", "📸 Take Photo"])
            with img_tab_upload:
                uploaded_image = st.file_uploader("Waste Image", type=["jpg", "jpeg", "png"])
            with img_tab_camera:
                camera_image = st.camera_input("Take a photo of the waste item")

            image_file = camera_image or uploaded_image

            question = st.text_input(
                "Your Question",
                value="Can I recycle this?",
                help="e.g. 'Where should I throw this?', 'Is this harmful to the environment?'",
            )

            if image_file:
                st.image(image_file, caption="Preview", use_column_width=True)

            predict_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            if not image_file:
                st.error("Please upload an image before analyzing.")
            elif not question.strip():
                st.error("Please enter a question.")
            else:
                try:
                    with st.spinner("Loading models and analyzing image..."):
                        cnn_model, class_names = load_cnn_model()
                        tokenizer, intent_model, id2label = load_intent_model()
                        knowledge = load_knowledge_base()
                        gemini_client = load_gemini_client()

                        image_path = save_uploaded_file(image_file)

                        result = generate_response(
                            image_path=image_path,
                            question=question,
                            cnn_model=cnn_model,
                            class_names=class_names,
                            tokenizer=tokenizer,
                            intent_model=intent_model,
                            id2label=id2label,
                            knowledge=knowledge,
                            gemini_client=gemini_client,
                        )

                    if gemini_client is None:
                        st.caption(
                            "ℹ️ AI-generated answers are off (no GEMINI_API_KEY configured) — "
                            "showing the standard knowledge-base response instead."
                        )

                    render_result_card(result)
                    # OPTIMIZATION: image is only needed for this single prediction,
                    # so remove the temp file immediately to avoid disk buildup.
                    cleanup_temp_file(image_path)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
        else:
            st.markdown(
                """
                <div class="placeholder-card">
                    <p>📤 Upload or take a photo and click <b>Analyze Image</b> to see results here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==========================================================================================
# PAGE: AI ASSISTANT
# ==========================================================================================
def page_ai_assistant():
    st.markdown("## 🤖 AI Assistant")
    st.caption("Upload or take a photo of the waste item once, then ask multiple questions by typing.")

    with st.container(border=True):
        st.markdown("#### Step 1 — Set the Waste Item")

        asst_tab_upload, asst_tab_camera = st.tabs(["📁 Upload", "📸 Take Photo"])
        with asst_tab_upload:
            uploaded_image = st.file_uploader(
                "Waste Image", type=["jpg", "jpeg", "png"], key="assistant_uploader"
            )
        with asst_tab_camera:
            camera_image = st.camera_input("Take a photo of the waste item", key="assistant_camera")

        image_file = camera_image or uploaded_image

        if image_file is not None:
            # OPTIMIZATION: remove the previously stored temp image (if any)
            # before saving the new one, so re-uploads/re-captures don't leak temp files.
            cleanup_temp_file(st.session_state["assistant_image_path"])
            image_path = save_uploaded_file(image_file)
            st.session_state["assistant_image_path"] = image_path
            st.session_state["assistant_image_name"] = getattr(image_file, "name", "captured_photo.jpg")

        if st.session_state["assistant_image_path"]:
            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(st.session_state["assistant_image_path"], use_column_width=True)
            with c2:
                st.success(f"Active image: **{st.session_state['assistant_image_name']}**")
                if st.button("🗑️ Clear Image & Chat"):
                    cleanup_temp_file(st.session_state["assistant_image_path"])
                    st.session_state["assistant_image_path"] = None
                    st.session_state["assistant_image_name"] = None
                    st.session_state["assistant_history"] = []
                    st.rerun()
        else:
            st.info("Upload or take a photo above to begin the conversation.")

    st.markdown("#### Step 2 — Ask a Question")

    question = None
    typed_question = st.text_input("Type your question", key="assistant_text_q")
    if st.button("Send", key="assistant_send_text", use_container_width=True):
        question = typed_question

    if question and question.strip():
        if not st.session_state["assistant_image_path"]:
            st.error("Please upload a waste image first (Step 1).")
        else:
            try:
                with st.spinner("Thinking..."):
                    cnn_model, class_names = load_cnn_model()
                    tokenizer, intent_model, id2label = load_intent_model()
                    knowledge = load_knowledge_base()
                    gemini_client = load_gemini_client()

                    result = generate_response(
                        image_path=st.session_state["assistant_image_path"],
                        question=question,
                        cnn_model=cnn_model,
                        class_names=class_names,
                        tokenizer=tokenizer,
                        intent_model=intent_model,
                        id2label=id2label,
                        knowledge=knowledge,
                        gemini_client=gemini_client,
                    )

                st.session_state["assistant_history"].append({"question": question, "result": result})

            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown("#### Conversation")
    if not st.session_state["assistant_history"]:
        st.markdown(
            """
            <div class="placeholder-card">
                <p>💬 Your conversation will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for turn in reversed(st.session_state["assistant_history"]):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                if turn["result"].get("Status") == "Error":
                    st.error(turn["result"].get("Message"))
                else:
                    r = turn["result"]
                    st.markdown(f"**{r['Detected Item'].title()}** ({r['Confidence']}% confidence)")
                    st.write(r["Response"])
                    with st.expander("View full details"):
                        st.markdown(display_prediction(r))


# ==========================================================================================
# PAGE: ABOUT
# ==========================================================================================
def page_about():
    st.markdown("## ℹ️ About EcoVision AI")

    with st.container(border=True):
        st.markdown(
            """
            **EcoVision AI** is an intelligent waste classification and recycling
            assistant. It combines computer vision and natural language understanding
            to help users correctly identify, sort, and dispose of waste items from a
            photo and a typed question.
            """
        )

    st.markdown("### 🧩 Technology Stack")
    tabs = st.tabs(["Vision", "Language", "AI Answers", "Interface"])

    with tabs[0]:
        st.markdown("**MobileNetV2** — Lightweight CNN fine-tuned for 10 waste categories.")
    with tabs[1]:
        st.markdown("**DistilBERT** — Fine-tuned intent classification for recycling-related questions.")
    with tabs[2]:
        st.markdown(
            "**Gemini 2.5 Flash** — Generates natural-language answers grounded in the "
            "knowledge base for the detected item, so responses address the user's exact "
            "question instead of a fixed set of intents. Falls back to the knowledge-base "
            "lookup automatically if no API key is configured."
        )
    with tabs[3]:
        st.markdown("**Streamlit** — Fully interactive, Python-only web dashboard (no backend server).")

    st.markdown("### 📈 Model Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Waste Classes", "10")
    c2.metric("Intent Categories", "Multiple")
    c3.metric("Architecture", "CNN + Transformer")

    with st.expander("🗂️ Supported Waste Categories"):
        st.write("Battery, Biological, Cardboard, Clothes, Glass, Metal, Paper, Plastic, Shoes, Trash")

    with st.expander("👨‍💻 Project Notes"):
        st.write(
            "Built as a fully Streamlit-native application — no FastAPI or external backend "
            "is used. All models are cached in-process via `st.cache_resource` for performance."
        )

    st.success("Thank you for using EcoVision AI ♻️")


# ==========================================================================================
# MAIN ROUTER
# ==========================================================================================
def main():
    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "📷 Image Prediction":
        page_image_prediction()
    elif page == "🤖 AI Assistant":
        page_ai_assistant()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()