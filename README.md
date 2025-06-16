# LLM Projects

A comprehensive collection of projects and examples demonstrating various applications of Large Language Models (LLMs) and related technologies.

## Table of Contents
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Components](#components)
  - [Basics](#basics)
  - [RAG Systems](#rag-systems)
  - [Agents](#agents)
  - [Embeddings](#embeddings)
  - [Chains](#chains)
  - [Chat History](#chat-history)
  - [Image Processing](#image-processing)
  - [Prompt Templates](#prompt-templates)
  - [Use Cases](#use-cases)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LLM-Projects
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Unix/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your-api-key-here
UNSTRUCTURED_API_KEY=your-unstructured-api-key-here  # Optional, for document processing
```

## Project Structure

The repository is organized into several key directories, each focusing on different aspects of LLM applications:

### Components

#### Basics
Location: `basics/`
- Basic LLM integration examples
- Files:
  - `gemma_demo.py`: Google's Gemma model integration
  - `openai_demo.py`: OpenAI API integration
  - `streamlit_demo.py`: Streamlit web application demo

Run a basic demo:
```bash
streamlit run basics/streamlit_demo.py
```

#### RAG Systems
Location: `rag/`
- Retrieval-Augmented Generation implementations
- Features:
  - PDF-based RAG
  - Document processing
  - History-aware RAG
  - Legal document analysis
- Key files:
  - `pdf_rag_demo.py`: Basic PDF RAG implementation
  - `multi_pdf_history_aware_rag.py`: Multi-PDF with history awareness
  - `Legal_bot.py`: Legal document analysis system

Run a RAG demo:
```bash
streamlit run rag/pdf_rag_demo.py
```

#### Agents
Location: `agents/`
- LLM Agent implementations
- Features:
  - Basic agent demonstrations
  - Transcript processing
- Key files:
  - `agent_demo.py`: Basic agent implementation
  - `transcript_to_article.py`: YouTube transcript processing

Run an agent demo:
```bash
streamlit run agents/agent_demo.py
```

#### Embeddings
Location: `embeddings/`
- Vector embeddings and similarity search
- Features:
  - Text embedding generation
  - Similarity comparison
  - Job search helper
- Key files:
  - `embeddings_demo.py`: Basic embeddings demonstration
  - `similarity_finder.py`: Text similarity comparison
  - `job_search_helper.py`: Job search using embeddings

Run an embeddings demo:
```bash
python embeddings/embeddings_demo.py
```

#### Chains
Location: `chains/`
- LangChain implementations
- Features:
  - Sequential chains
  - Complex processing pipelines
- Key files:
  - `sequential_chain.py`: Sequential processing example

Run a chain demo:
```bash
streamlit run chains/sequential_chain.py
```

#### Chat History
Location: `chathistory/`
- Chat history management and processing
- Features:
  - Conversation tracking
  - History-aware responses

#### Image Processing
Location: `imageprocessing/`
- Image processing and analysis
- Features:
  - Image analysis using LLMs
  - KYC use case implementation
- Key files:
  - `images_demo.py`: Basic image processing
  - `kyc_usecase.py`: KYC verification system
  - `streamlit_images_demo.py`: Web interface for image processing

Run an image processing demo:
```bash
streamlit run imageprocessing/streamlit_images_demo.py
```

#### Prompt Templates
Location: `prompttemplates/`
- Reusable prompt templates
- Features:
  - Custom prompt patterns
  - Template management
- Key files:
  - `prompttemplate_demo.py`: Prompt template demonstration

Run a prompt template demo:
```bash
streamlit run prompttemplates/prompttemplate_demo.py
```

#### Use Cases
Location: `use_case/`
- Specific application implementations
- Features:
  - Multi-format RAG
  - Meal planning
  - Essay writing
  - Chat systems
- Key files:
  - `multi_format_rag.py`: Multi-format document processing
  - `meal_planner.py`: AI-powered meal planning
  - `agentic_essay_writer.py`: Automated essay writing
  - `chat_with_me.py`: Custom chat implementation

Run a use case demo:
```bash
streamlit run use_case/meal_planner.py
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for their API and models
- Google for the Gemma model
- The LangChain team for their excellent framework
- The open-source community for various tools and libraries used in this project 