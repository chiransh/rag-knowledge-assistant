"""
RAG Knowledge Assistant

A Retrieval Augmented Generation application backed by txtai.
Supports both vector RAG and graph RAG over a user-built knowledge index.
"""

import os
import re

from glob import glob
from io import BytesIO
from uuid import UUID

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from txtai import Embeddings, LLM, RAG
from txtai.pipeline import Textractor


class AutoId:
    """
    Helper methods to detect txtai auto-generated IDs (UUID or numeric).
    Used to distinguish user-supplied IDs from system-generated ones
    when assigning LLM topics to graph nodes.
    """


class GraphContext:
    """
    Builds retrieval context for Graph RAG queries.

    Supports three query modes:
      - gq: <query>         — graph query expansion from a vector search seed
      - A -> B -> C         — explicit concept path traversal
      - A -> B gq: <query>  — path traversal combined with graph query
    """


class Application:
    """
    Main RAG application.

    Manages the embeddings index, LLM, RAG pipeline, and Streamlit UI.
    Handles both vector and graph-based retrieval depending on the query type.
    """


@st.cache_resource(show_spinner="Initializing models and database...")
def create():
    """
    Creates and caches the Application instance across Streamlit reruns.
    """


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    st.set_page_config(
        page_title="RAG Knowledge Assistant",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title(os.environ.get("TITLE", "🧠 RAG Knowledge Assistant"))

    app = create()
    app.run()
