"""
utils.py
--------
EcoVision AI - Core Helper Functions

This module contains ONLY helper/logic functions:
    - Cached model loaders (MobileNetV2, DistilBERT, Knowledge Base)
    - Inference wrappers (image classification, intent detection)
    - Response generation and formatting logic

No Streamlit UI widgets are used here. Pages import these functions and
handle all rendering (st.write, st.image, etc.) themselves.

AI logic (preprocessing, inference calls, response construction) is reused
as-is from the original notebooks:
    - 04_Model_Evaluation.ipynb      -> IMG_SIZE = 224 (MobileNetV2 input size)
    - 06_Intent_Detection.ipynb      -> DistilBERT tokenizer/model + id2label
    - 07_Response_Generator.ipynb    -> predict_image / predict_intent /
                                         get_matching_class / generate_response
"""

import os
import json
import logging
import re

import numpy as np
import torch
from PIL import Image

import streamlit as st

# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("EcoVisionAI")


# ------------------------------------------------------------------
# Path Configuration
# (Matches the production folder structure agreed earlier)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2", "best_mobilenetv2.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "data", "class_names.json")

INTENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert")
LABEL_MAPPING_PATH = os.path.join(INTENT_MODEL_PATH, "label_mapping.json")

KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.py")

# MobileNetV2 input size, as used during training / evaluation
# (04_Model_Evaluation.ipynb)
IMG_SIZE = 224


# ==========================================================================================
# MODEL LOADERS  (cached with @st.cache_resource -> loaded once per server process)
# ==========================================================================================

@st.cache_resource(show_spinner="Loading Waste Classification Model (MobileNetV2)...")
def load_cnn_model():
    """
    Loads the trained MobileNetV2 waste classification model along with
    its associated class names.

    Returns:
        tuple: (cnn_model, class_names)
            cnn_model (tf.keras.Model): Loaded Keras model.
            class_names (list[str]): Ordered list of waste class labels.

    Raises:
        FileNotFoundError: If the model or class names file is missing.
        RuntimeError: If the model fails to load.
    """
    try:
        import tensorflow as tf

        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(f"CNN model not found at: {CNN_MODEL_PATH}")

        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"class_names.json not found at: {CLASS_NAMES_PATH}")

        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)

        logger.info("MobileNetV2 model loaded successfully.")
        return cnn_model, class_names

    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
        raise
    except Exception as e:
        logger.exception("Failed to load MobileNetV2 model.")
        raise RuntimeError(f"Error loading CNN model: {e}") from e



@st.cache_resource(show_spinner="Loading Intent Detection Model (DistilBERT)...")
def load_intent_model():
    """
    Loads the fine-tuned DistilBERT intent detection model, its tokenizer,
    and the id2label / label2id mapping.

    Returns:
        tuple: (tokenizer, intent_model, id2label)
            tokenizer (AutoTokenizer): DistilBERT tokenizer.
            intent_model (AutoModelForSequenceClassification): Loaded model.
            id2label (dict): Mapping of prediction index -> intent label.

    Raises:
        FileNotFoundError: If the model directory or label mapping is missing.
        RuntimeError: If the model fails to load.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not os.path.exists(INTENT_MODEL_PATH):
            raise FileNotFoundError(f"Intent model directory not found at: {INTENT_MODEL_PATH}")

        if not os.path.exists(LABEL_MAPPING_PATH):
            raise FileNotFoundError(f"label_mapping.json not found at: {LABEL_MAPPING_PATH}")

        tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
        intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)

        # BUGFIX: without eval(), dropout layers stay active during inference,
        # producing non-deterministic intent predictions for the same question.
        intent_model.eval()

        with open(LABEL_MAPPING_PATH, "r") as f:
            mapping = json.load(f)

        id2label = mapping["id2label"]

        logger.info("DistilBERT intent detection model loaded successfully.")
        return tokenizer, intent_model, id2label

    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
        raise
    except Exception as e:
        logger.exception("Failed to load DistilBERT intent model.")
        raise RuntimeError(f"Error loading intent model: {e}") from e


@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_knowledge_base():
    """
    Loads the waste knowledge base dictionary from data/knowledge_base.py.

    Returns:
        dict: Knowledge base mapping waste categories to disposal,
              recycling, and environmental information.

    Raises:
        FileNotFoundError: If knowledge_base.py cannot be located.
        RuntimeError: If the module fails to load.
    """
    try:
        import importlib.util

        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            raise FileNotFoundError(f"knowledge_base.py not found at: {KNOWLEDGE_BASE_PATH}")

        spec = importlib.util.spec_from_file_location("knowledge_base", KNOWLEDGE_BASE_PATH)
        knowledge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(knowledge_module)

        knowledge = knowledge_module.knowledge

        logger.info(f"Knowledge base loaded successfully. Total classes: {len(knowledge)}")
        return knowledge

    except FileNotFoundError as e:
        logger.error(f"Knowledge base file missing: {e}")
        raise
    except Exception as e:
        logger.exception("Failed to load knowledge base.")
        raise RuntimeError(f"Error loading knowledge base: {e}") from e


# ==========================================================================================
# INFERENCE FUNCTIONS
# ==========================================================================================

def predict_image(image_path, cnn_model, class_names):
    """
    Predicts the waste category of an image using the MobileNetV2 model.

    Reuses the exact preprocessing/inference logic from
    07_Response_Generator.ipynb (IMG_SIZE = 224, normalized to [0, 1]).

    Args:
        image_path (str): Path to the image file (or a file-like object
            accepted by PIL.Image.open).
        cnn_model: Loaded Keras MobileNetV2 model (from load_cnn_model()).
        class_names (list[str]): Ordered class labels (from load_cnn_model()).

    Returns:
        tuple: (predicted_class (str), confidence (float))

    Raises:
        RuntimeError: If image loading or prediction fails.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = cnn_model.predict(image, verbose=0)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = class_names[class_index]

        logger.info(f"Image predicted as '{predicted_class}' ({confidence * 100:.2f}%).")
        return predicted_class, confidence

    except Exception as e:
        logger.exception("Image prediction failed.")
        raise RuntimeError(f"Error predicting image: {e}") from e


# ==========================================================================================
# INTENT KEYWORD SAFETY NET
# ==========================================================================================
# With only ~100 training examples per intent, the DistilBERT classifier can latch
# onto sentence *shape* (e.g. "Can I ___ this?") rather than the specific verb,
# misfiring on short/uncommon phrasings (e.g. "Can I dispose this?" -> Reuse).
# These keyword patterns are near-unambiguous domain terms; if one is present and
# the model isn't confident, we trust the keyword over the model.
# Ordered so more specific/less overlapping terms are checked first.
_INTENT_KEYWORD_PATTERNS = [
    ("Recycling", [r"\brecycl\w*"]),
    ("Decomposition", [r"\bdecompos\w*", r"\bbiodegrad\w*", r"\bdegrad\w*",
                        r"\bbreak(s)?\s+down\b", r"\bdisintegrat\w*", r"\brot(s|ted|ting)?\b"]),
    ("Reuse", [r"\breus\w*", r"\brepurpos\w*", r"\bupcycl\w*", r"\bsecond\s+life\b"]),
    ("Material", [r"\bmaterial\b", r"\bmade\s+(of|from|out\s+of)\b", r"\bcomposition\b",
                   r"\bconstructed\s+from\b"]),
    ("Environment", [r"\benvironment\w*", r"\beco[\s-]?friendly\b", r"\bpollut\w*",
                      r"\bcarbon\s+footprint\b", r"\bwildlife\b", r"\bsustainab\w*"]),
    ("Disposal", [r"\bdispos\w*", r"\bthrow\s*(away|out)?\b", r"\bdiscard\w*",
                   r"\bget\s+rid\s+of\b", r"\bwhich\s+bin\b", r"\btrash\b", r"\bgarbage\b"]),
]


def _match_intent_keyword(question):
    """
    Checks the question for an unambiguous domain keyword and returns the
    matching intent, or None if no keyword pattern matches.
    """
    text = question.lower()
    for intent, patterns in _INTENT_KEYWORD_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return intent
    return None


def predict_intent(question, tokenizer, intent_model, id2label, confidence_threshold=0.85):
    """
    Predicts the user's intent from a text question using DistilBERT.

    Reuses the exact inference logic from 07_Response_Generator.ipynb, with an
    added keyword-based safety net: if the model's confidence is below
    `confidence_threshold` AND the question contains an unambiguous domain
    keyword that points to a *different* intent, the keyword wins. This
    guards against the small-dataset generalization gaps described above
    without discarding the model for phrasing it already handles well.

    Args:
        question (str): The user's transcribed/typed question.
        tokenizer: DistilBERT tokenizer (from load_intent_model()).
        intent_model: DistilBERT sequence classification model.
        id2label (dict): Mapping of prediction index -> intent label.
        confidence_threshold (float): Softmax confidence below which the
            keyword safety net is allowed to override the model. Defaults
            to 0.85 (favors the keyword unless the model is very sure).

    Returns:
        str: Predicted intent label.

    Raises:
        RuntimeError: If tokenization or inference fails.
    """
    try:
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # DistilBERT does not accept token_type_ids
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
           outputs = intent_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        confidence = confidence.item()
        prediction = prediction.item()
        model_intent = id2label[str(prediction)]

        keyword_intent = _match_intent_keyword(question)

        if keyword_intent and keyword_intent != model_intent and confidence < confidence_threshold:
            logger.info(
                f"Intent override: model predicted '{model_intent}' ({confidence:.2f} confidence) "
                f"for '{question}', keyword safety net overrode to '{keyword_intent}'."
            )
            intent = keyword_intent
        else:
            intent = model_intent

        logger.info(f"Predicted intent: '{intent}' for question: '{question}'")
        return intent

    except Exception as e:
        logger.exception("Intent prediction failed.")
        raise RuntimeError(f"Error predicting intent: {e}") from e



# ==========================================================================================
# GEMINI AI RESPONSE GENERATION  (reused from 08_HuggingFace_AI_Response.ipynb)
# ==========================================================================================
GEMINI_MODEL_NAME = "gemini-2.5-flash"


def _get_gemini_api_key():
    """
    Retrieves the Gemini API key from Streamlit secrets (preferred) or an
    environment variable. The key must NEVER be hardcoded in source.

    Setup (pick one):
      - .streamlit/secrets.toml:   GEMINI_API_KEY = "your-key-here"
      - Environment variable:      export GEMINI_API_KEY="your-key-here"
    """
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_gemini_client():
    """
    Initializes and caches the Gemini API client.

    Reuses the exact client setup from 08_HuggingFace_AI_Response.ipynb.
    Unlike the notebook, this does NOT hardcode the API key - it is read
    from Streamlit secrets / environment variables instead.

    Returns:
        genai.Client | None: The connected client, or None if no API key
        is configured (callers should fall back to the knowledge-base-only
        response instead of crashing the app).
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY not found (checked st.secrets and environment). "
            "AI-generated responses are disabled; falling back to knowledge-base lookup."
        )
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client connected successfully.")
        return client
    except Exception as e:
        logger.exception("Failed to initialize Gemini client.")
        return None


def build_ai_prompt(item, question, knowledge):
    """
    Builds the Gemini prompt from the detected item's knowledge base entry.

    Reuses the exact prompt template from 08_HuggingFace_AI_Response.ipynb.

    Args:
        item (str): Matched knowledge base class name (e.g. "Plastic").
        question (str): The user's question (typed or transcribed).
        knowledge (dict): Knowledge base dictionary.

    Returns:
        str: The fully formatted prompt.
    """
    info = knowledge[item]

    prompt = f"""
You are an intelligent waste management assistant.

Detected Waste Item:
{item}

Knowledge:

Material:
{info["Material"]}

Recyclable:
{info["Recyclable"]}

Recommended Bin:
{info["Bin"]}

Environmental Impact:
{info["Environment"]}

Reuse:
{info["Reuse"]}

Decomposition:
{info["Decomposition"]}

Tips:
{", ".join(info["Tips"])}

Warnings:
{", ".join(info["Warnings"])}

User Question:

{question}

Instructions:

1. Answer naturally.
2. Use ONLY the knowledge above.
3. Do not invent facts.
4. Answer in less than 120 words.
5. Be friendly.
"""
    return prompt


def ask_gemini(prompt, client, max_retries=3):
    """
    Calls the Gemini API with the built prompt.

    Reuses the exact rate-limit retry logic from
    08_HuggingFace_AI_Response.ipynb (Cell 12), bounded to `max_retries` so
    a persistent outage/quota issue can't hang the Streamlit app forever.

    Args:
        prompt (str): The prompt built by build_ai_prompt().
        client: Connected Gemini client (from load_gemini_client()).
        max_retries (int): Max number of retries on HTTP 429 (rate limit).

    Returns:
        str: The generated answer text.

    Raises:
        RuntimeError: If the API call fails for a non-rate-limit reason, or
            the rate limit persists past max_retries.
    """
    from google.genai.errors import ClientError
    import time

    attempts = 0
    while attempts < max_retries:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt
            )
            return response.text
        except ClientError as e:
            if "429" in str(e):
                attempts += 1
                logger.warning(f"Gemini rate limit hit, retrying ({attempts}/{max_retries})...")
                time.sleep(10)
            else:
                logger.exception("Gemini API call failed.")
                raise RuntimeError(f"Gemini API error: {e}") from e
        except Exception as e:
            logger.exception("Gemini API call failed.")
            raise RuntimeError(f"Gemini API error: {e}") from e

    raise RuntimeError("Gemini API rate limit exceeded after multiple retries.")


# ==========================================================================================
# KNOWLEDGE BASE MATCHING  (internal helper, reused from 07_Response_Generator.ipynb)
# ==========================================================================================

def _get_matching_class(predicted_class, knowledge):
    """
    Matches a predicted waste class (or its alias) to a knowledge base key.

    Args:
        predicted_class (str): Class name predicted by the CNN model.
        knowledge (dict): Knowledge base (from load_knowledge_base()).

    Returns:
        str | None: Matched knowledge base key, or None if no match found.
    """
    predicted_class = predicted_class.strip().lower()

    # Pass 1: exact class name or alias match (original notebook logic, unchanged)
    for waste_class, info in knowledge.items():

        if waste_class.lower() == predicted_class:
            return waste_class

        if "Aliases" in info:
            aliases = [alias.lower() for alias in info["Aliases"]]
            if predicted_class in aliases:
                return waste_class

    # BUGFIX / FALLBACK: some CNN classes are generic (e.g. "glass") while the
    # knowledge base only has specific variants (e.g. "Brown Glass",
    # "White Glass", "Green Glass"). Without this fallback, every prediction
    # of such a class returns "No knowledge found" even though a reasonable
    # match exists. Match if the predicted word appears as a whole word
    # inside a knowledge base key (e.g. "glass" -> "brown glass").
    for waste_class in knowledge.keys():
        waste_class_words = waste_class.lower().split()
        if predicted_class in waste_class_words:
            logger.info(
                f"'{predicted_class}' matched to '{waste_class}' via fallback "
                f"word matching (no exact class/alias match found)."
            )
            return waste_class

    return None


# ==========================================================================================
# RESPONSE GENERATION
# ==========================================================================================

def generate_response(image_path, question, cnn_model, class_names,
                       tokenizer, intent_model, id2label, knowledge,
                       gemini_client=None):
    """
    Full EcoVision AI pipeline: classifies the waste image, detects the
    user's intent (for display only), and generates a natural-language
    answer to the user's exact question.

    If a Gemini client is provided, the answer is generated by Gemini using
    the knowledge base as grounding context (08_HuggingFace_AI_Response.ipynb),
    so the response directly addresses whatever the user actually asked -
    instead of being limited to one of the 6 fixed intent buckets.

    If no Gemini client is available (e.g. GEMINI_API_KEY not configured),
    this transparently falls back to the original fixed-field lookup
    (07_Response_Generator.ipynb) so the app keeps working either way.

    Args:
        image_path (str): Path to the uploaded/captured waste image.
        question (str): User's question (typed or transcribed from speech).
        cnn_model: Loaded MobileNetV2 model.
        class_names (list[str]): CNN class labels.
        tokenizer: DistilBERT tokenizer.
        intent_model: DistilBERT intent model.
        id2label (dict): Intent index -> label mapping.
        knowledge (dict): Knowledge base dictionary.
        gemini_client: Connected Gemini client from load_gemini_client(),
            or None to use the fixed-field fallback.

    Returns:
        dict: Structured result containing Status and, on success, the
              full set of knowledge base fields for the detected item.
    """
    try:
        # Step 1: Predict Waste Class
        waste_class, confidence = predict_image(image_path, cnn_model, class_names)

        # Step 2: Predict Intent (kept for the "Intent detected" UI label;
        # no longer required for the answer itself when Gemini is available)
        intent = predict_intent(question, tokenizer, intent_model, id2label)

        # Step 3: Match Waste Class Against Knowledge Base
        matched_class = _get_matching_class(waste_class, knowledge)

        if matched_class is None:
            logger.warning(f"No knowledge base match found for '{waste_class}'.")
            return {
                "Status": "Error",
                "Message": f"No knowledge found for '{waste_class}'."
            }

        waste_info = knowledge[matched_class]

        # Step 4: Generate the answer - Gemini (grounded on the knowledge
        # base) when available, otherwise the original fixed-field lookup.
        if gemini_client is not None:
            try:
                prompt = build_ai_prompt(matched_class, question, knowledge)
                response_text = ask_gemini(prompt, gemini_client)
            except RuntimeError:
                logger.exception("Gemini generation failed; falling back to fixed-field lookup.")
                response_text = None
        else:
            response_text = None

        if response_text is None:
            # BUGFIX: exact-case lookup could KeyError if the intent model's
            # label casing (e.g. "recycling") doesn't exactly match the
            # knowledge base field casing (e.g. "Recycling"). Resolve
            # case-insensitively instead of crashing.
            matched_field = next(
                (field for field in waste_info if field.lower() == intent.lower()),
                None
            )
            response_text = (
                waste_info[matched_field] if matched_field
                else waste_info.get("ConfidenceMessage", "")
            )

        result = {
            "Status": "Success",
            "Detected Item": matched_class,
            "Confidence": round(confidence * 100, 2),
            "Intent": intent,
            "Recyclable": waste_info["Recyclable"],
            "Recommended Bin": waste_info["Bin"],
            "Response": response_text,
            "Material": waste_info["Material"],
            "Environmental Impact": waste_info["Environment"],
            "Decomposition": waste_info["Decomposition"],
            "Confidence Message": waste_info["ConfidenceMessage"],
            "Tips": waste_info["Tips"],
            "Warnings": waste_info["Warnings"],
            "Common Uses": waste_info["CommonUses"],
            "Environmental Score": waste_info["EnvironmentalScore"],
            "Carbon Impact": waste_info["CarbonImpact"],
            "Interesting Fact": waste_info["InterestingFact"],
        }

        logger.info(f"Response generated successfully for '{matched_class}' / intent '{intent}'.")
        return result

    except RuntimeError:
        # Already logged in predict_image / predict_intent - re-raise for caller to handle
        raise
    except Exception as e:
        logger.exception("Response generation failed.")
        raise RuntimeError(f"Error generating response: {e}") from e


# ==========================================================================================
# DISPLAY FORMATTING  (data preparation only - NO Streamlit widgets)
# ==========================================================================================

def display_prediction(result):
    """
    Formats a generate_response() result into a clean, display-ready
    markdown string. Contains NO Streamlit calls - pages are responsible
    for rendering the returned string (e.g., via st.markdown).

    Args:
        result (dict): Output from generate_response().

    Returns:
        str: Formatted markdown text ready for display.
    """
    try:
        if result.get("Status") == "Error":
            return f"**⚠️ {result.get('Message', 'An unknown error occurred.')}**"

        lines = [
            "## 🌍 EcoVision AI - Result",
            f"**Detected Item:** {result['Detected Item']}",
            f"**Confidence:** {result['Confidence']}%",
            f"**Intent:** {result['Intent']}",
            f"**Recyclable:** {'✅ Yes' if result['Recyclable'] else '❌ No'}",
            f"**Recommended Bin:** {result['Recommended Bin']}",
            "",
            f"### Recommendation\n{result['Response']}",
            f"### Material\n{result['Material']}",
            f"### Environmental Impact\n{result['Environmental Impact']}",
            f"### Decomposition\n{result['Decomposition']}",
            f"### Confidence Message\n{result['Confidence Message']}",
            "### Tips",
            "\n".join(f"- {tip}" for tip in result["Tips"]),
            "### Common Uses",
            "\n".join(f"- {use}" for use in result["Common Uses"]),
            f"### Environmental Score\n{result['Environmental Score']} / 10",
            f"### Carbon Impact\n{result['Carbon Impact']}",
            f"### Interesting Fact\n{result['Interesting Fact']}",
        ]

        return "\n\n".join(lines)

    except KeyError as e:
        logger.exception("Malformed result passed to display_prediction.")
        return f"**⚠️ Unable to format result - missing field: {e}**"
    except Exception as e:
        logger.exception("Failed to format prediction result.")
        return f"**⚠️ Unable to display result: {e}**"
