# 🧠 RAG Knowledge Assistant

A Retrieval Augmented Generation (RAG) application that lets you ask natural language questions over any knowledge base you build.

## What this does

RAG Knowledge Assistant helps you get accurate, context-grounded answers from an LLM by limiting what the model can respond with to only what exists in your indexed knowledge base. This avoids hallucination and keeps answers factually relevant to your data.

The application supports two modes of retrieval:

- **Vector RAG** — finds the most semantically relevant documents to a question and supplies them as context to the LLM
- **Graph RAG** — traverses a knowledge graph built from your documents to reason across connected concepts

## Motivation

Large language models are powerful but unreliable when asked about specific, private, or domain-specific knowledge. This project explores a practical pattern for grounding LLM responses in real, user-controlled data — whether that data is web pages, uploaded files, or custom text notes.

## Planned features

- Upload URLs, local files, or raw text to build a live knowledge index
- Ask questions and get answers backed by your own documents
- Visualize how the model reasoned over connected concepts via graph views
- Configurable LLM and embedding model backends
- Docker support for easy local or cloud deployment

## Status

> 🚧 Project in active development. Follow along as the codebase is built up incrementally.
