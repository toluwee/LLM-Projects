# Building AI Agents with LangChain: A Comprehensive Tutorial

This tutorial will walk you through building an AI agent using LangChain, explaining each component and suggesting alternatives along the way.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Understanding the Components](#understanding-the-components)
5. [Building the Agent](#building-the-agent)
6. [Creating the User Interface](#creating-the-user-interface)
7. [Best Practices and Tips](#best-practices-and-tips)
8. [Next Steps](#next-steps)

## Introduction

In this tutorial, we'll explore how to build an AI agent that can use tools like Wikipedia and DuckDuckGo search to answer questions and complete tasks. The agent we'll build is based on the ReAct (Reasoning and Acting) framework, which allows it to break down complex tasks into manageable steps.

## Prerequisites

Before starting, make sure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- A DeepSeek API key (or alternative LLM API key)

## Setting Up the Environment

First, let's install the required packages. Create a new virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install langchain langchain-openai streamlit wikipedia python-dotenv
```

## Understanding the Components

Let's break down the main components of our agent implementation:

### 1. Language Model Setup

```python
import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.0,
    max_tokens=1000,
    base_url="https://api.deepseek.com"
)
```

**Alternative Language Models:**
- OpenAI's GPT models:
  ```python
  llm = ChatOpenAI(model="gpt-4")
  ```
- Anthropic's Claude:
  ```python
  from langchain_anthropic import ChatAnthropic
  llm = ChatAnthropic(model="claude-3-opus-20240229")
  ```
- Local models using Ollama:
  ```python
  from langchain_community.llms import Ollama
  llm = Ollama(model="llama2")
  ```

### 2. Tools Setup

```python
from langchain_community.agent_toolkits.load_tools import load_tools

tools = load_tools([
    "wikipedia",
    "ddg-search",
])
```

**Alternative Tools:**
- `serpapi`: Google search integration
- `wolfram-alpha`: Mathematical computations
- `python_repl`: Run Python code
- `requests`: Make HTTP requests
- Custom tools you create yourself

### 3. Agent Creation

```python
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)
```

**Alternative Agent Types:**
1. OpenAI Functions Agent:
   ```python
   from langchain.agents import create_openai_functions_agent
   agent = create_openai_functions_agent(llm, tools, prompt)
   ```

2. Structured Chat Agent:
   ```python
   from langchain.agents import create_structured_chat_agent
   agent = create_structured_chat_agent(llm, tools, prompt)
   ```

### 4. User Interface

```python
import streamlit as st

st.title("AI Agent")
task = st.text_input("Assign me a task")

if task:
    try:
        response = agent_executor.invoke({"input": task})
        st.write(response["output"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try again in a few moments.")
```

**Alternative UI Options:**
1. Gradio: Simpler alternative to Streamlit
2. FastAPI: For creating REST APIs
3. Flask: For traditional web applications
4. Command-line interface: For simpler applications

## Best Practices and Tips

1. **Error Handling**
   - Always implement proper error handling
   - Use try-except blocks for API calls
   - Provide meaningful error messages to users

2. **API Key Management**
   - Never hardcode API keys
   - Use environment variables
   - Consider using a secrets management service

3. **Performance Optimization**
   - Set appropriate temperature values
   - Limit max_tokens to control costs
   - Use caching when appropriate

4. **Security**
   - Validate user input
   - Implement rate limiting
   - Monitor API usage

## Next Steps

1. **Enhance the Agent**
   - Add more tools
   - Implement memory for context
   - Add streaming responses

2. **Improve the UI**
   - Add loading indicators
   - Implement chat history
   - Add file upload capabilities

3. **Advanced Features**
   - Implement multi-agent systems
   - Add custom tool creation
   - Implement agent memory

4. **Testing and Deployment**
   - Write unit tests
   - Set up CI/CD
   - Deploy to cloud platforms

## Example Usage

Here's how to use the agent:

1. Set up your environment variables:
   ```bash
   export DEEPSEEK_API_KEY="your-api-key"
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run your_agent_file.py
   ```

3. Open your browser and interact with the agent through the web interface.

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Verify your API key is correctly set
   - Check if the API key has the necessary permissions
   - Ensure the API key is not expired

2. **Tool Loading Errors**
   - Verify all required packages are installed
   - Check internet connectivity
   - Ensure tool configurations are correct

3. **Memory Issues**
   - Monitor token usage
   - Implement proper error handling
   - Use appropriate max_tokens settings

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DeepSeek API Documentation](https://platform.deepseek.com/docs)
- [LangChain Agents Guide](https://python.langchain.com/docs/modules/agents/) 