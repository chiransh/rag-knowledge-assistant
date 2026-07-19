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

    @staticmethod
    def valid(uid):
        """
        Checks if uid is a valid auto id (UUID or numeric).

        Args:
            uid: input id to check

        Returns:
            True if this is an auto-generated id, False otherwise
        """

        # Check if this is a UUID
        try:
            return UUID(str(uid))
        except ValueError:
            pass

        # Return True if numeric, False otherwise
        return isinstance(uid, int) or (isinstance(uid, str) and uid.isdigit())


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

    def __init__(self):
        # Textractor instance — lazy loaded on first file/URL upload
        self.textractor = None

        # Load LLM from env or fall back to default open-weight model
        self.llm = LLM(os.environ.get("LLM", "Qwen/Qwen3-4B-Instruct-2507"))

        # Load or create the embeddings index
        self.embeddings = self.load()

        # Number of context records passed to the LLM per query
        self.context = int(os.environ.get("CONTEXT", 10))

        # Prompt template — constrains the LLM to only use supplied context
        template = """
Answer the following question using only the context below. Only include information
specifically discussed.

question: {question}
context: {context} """

        # Build the RAG pipeline with the loaded embeddings and LLM
        self.rag = RAG(
            self.embeddings,
            self.llm,
            system="You are a friendly assistant. You answer questions from users.",
            template=template,
            context=self.context,
        )

    def load(self):
        """
        Loads an existing embeddings index or creates a new one if none is found.

        Resolution order:
          1. Load from local path if EMBEDDINGS points to an existing directory
          2. Pull from HuggingFace Hub if EMBEDDINGS is a hub repo name
          3. Fall back to a fresh empty index via create()

        Also indexes an optional local DATA directory on startup and persists
        the updated index if PERSIST is set.

        Returns:
            Embeddings: a ready-to-query embeddings index
        """

        embeddings = None
        data = os.environ.get("DATA")
        database = os.environ.get("EMBEDDINGS", "neuml/txtai-wikipedia-slim")

        if database:
            embeddings = Embeddings()

            if embeddings.exists(database):
                # Load from local path
                embeddings.load(database)
            elif not os.path.isabs(database) and embeddings.exists(
                cloud={"provider": "huggingface-hub", "container": database}
            ):
                # Pull from HuggingFace Hub
                embeddings.load(provider="huggingface-hub", container=database)
            else:
                embeddings = None

        # Fall back to a new empty index if nothing was found
        embeddings = embeddings if embeddings else self.create()

        # Index a local data directory if provided
        if data:
            embeddings.upsert(self.stream(data))
            self.infertopics(embeddings, 0)
            self.persist(embeddings)

        return embeddings

    def create(self):
        """
        Creates a new empty embeddings index with graph support enabled.

        Uses:
          - intfloat/e5-large for dense vector embeddings
          - UUID5 auto-ids for content-addressed deduplication
          - E5 query/passage instruction prefixes
          - Approximate-off graph for precise similarity edges

        Returns:
            Embeddings: a fresh, empty embeddings index ready for upsert
        """

        return Embeddings(
            autoid="uuid5",
            path="intfloat/e5-large",
            instructions={"query": "query: ", "data": "passage: "},
            content=True,
            graph={"approximate": False, "minscore": 0.7},
        )

    def stream(self, data):
        """
        Walks a directory recursively and streams extracted text sections
        from every file found, ready for upsert into the embeddings index.

        Args:
            data: path to the root data directory

        Yields:
            extracted text sections from each file
        """

        for sections in self.extract(glob(f"{data}/**/*", recursive=True)):
            yield from sections

    def extract(self, inputs):
        """
        Runs the Textractor pipeline over one or more inputs (file paths or URLs)
        and returns extracted paragraph-level sections.

        Textractor is lazy-initialised on first call so the heavy import and
        Java runtime only load when actually needed.

        Args:
            inputs: a single input or list of file paths / URLs

        Returns:
            list of extracted text sections
        """

        if not self.textractor:
            self.textractor = Textractor(
                paragraphs=True,
                backend=os.environ.get("TEXTBACKEND", "available"),
                safeopen=os.environ.get("SAFEOPEN", "True").lower() in ("true", "1"),
            )

        return self.textractor(inputs)

    def addurl(self, url):
        """
        Adds content from a URL or local file path into the live embeddings index.

        Workflow:
          1. Record how many entries exist before ingestion
          2. Extract and upsert the new content
          3. Infer LLM-generated topics for any new graph nodes
          4. Persist the updated index to disk (if PERSIST is set)

        Args:
            url: a URL string or local file path to ingest
        """

        # Snapshot current count so infertopics only processes new entries
        start = self.embeddings.count()

        self.embeddings.upsert(self.extract(url))
        self.infertopics(self.embeddings, start)
        self.persist(self.embeddings)

    def persist(self, embeddings):
        """
        Saves the embeddings index to disk if the PERSIST env variable is set.

        This is a no-op when PERSIST is unset, allowing the app to run in
        fully in-memory mode without any filesystem side effects.

        Args:
            embeddings: the embeddings instance to save
        """

        persist = os.environ.get("PERSIST")
        if persist:
            embeddings.save(persist)

    def topics(self, embeddings, batch):
        """
        Generates a short topic label for each (uid, text) pair in the batch
        using the LLM, then writes the result directly onto each graph node.

        A single prompt is built per entry and the whole batch is sent to the
        LLM in one call to minimise round-trips. An optional TOPICSBATCH env
        variable controls the LLM's internal batch size for memory-constrained
        environments.

        Args:
            embeddings: the embeddings instance whose graph nodes will be labelled
            batch: list of (node_id, text) tuples to generate topics for
        """

        prompt = """
Create a simple, concise topic for the following text. Only return the topic name.

Text:
{text}"""

        # Build one prompt message per entry in the batch
        prompts = []
        for uid, text in batch:
            text = text if re.search(r"\w+", text) else uid
            prompts.append([{"role": "user", "content": prompt.format(text=text)}])

        # Respect optional batch size cap for memory-constrained setups
        topicsbatch = os.environ.get("TOPICSBATCH")
        kwargs = {"batch_size": int(topicsbatch)} if topicsbatch else {}

        # Run all prompts through the LLM and assign results to graph nodes
        for x, topic in enumerate(
            self.llm(
                prompts,
                maxlength=int(os.environ.get("MAXLENGTH", 2048)),
                stripthink=os.environ.get("STRIPTHINK", "false").lower() in ("true", "1"),
                **kwargs,
            )
        ):
            uid = batch[x][0]
            embeddings.graph.addattribute(uid, "topic", topic)

            # Register the topic in the graph's topic index
            topics = embeddings.graph.topics
            if topics:
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(uid)


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
