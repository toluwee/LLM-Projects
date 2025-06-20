import os
from typing import Dict, Any, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
import time
import streamlit as st

# Prompts
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------
"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    topic: Annotated[str, "The essay topic"]
    outline: Annotated[str, "The essay outline"]
    research: Annotated[str, "The research content"]
    essay: Annotated[str, "The current essay"]
    critique: Annotated[str, "The critique of the essay"]
    iteration: Annotated[int, "The current iteration number"]
    max_iterations: Annotated[int, "Maximum number of revision iterations"]
    word_count_limit: Annotated[int, "Maximum word count for the essay"]

# Initialize the LLM
def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# Initialize the search tool
def get_search_tool():
    return GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID")
    )

def search_with_retry(query: str, max_retries: int = 3, delay: int = 2) -> str:
    """Perform search with retry logic for rate limits."""
    search = get_search_tool()
    for attempt in range(max_retries):
        try:
            results = search.results(query, num_results=3)  # Get top 3 results
            if results:
                # Format results with titles and snippets
                formatted_results = []
                for result in results:
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', 'No description')
                    formatted_results.append(f"Title: {title}\nSummary: {snippet}\n")
                return "\n".join(formatted_results)
            return "No relevant information found."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
                continue
            return f"Error performing search: {str(e)}"
    return "Failed to retrieve search results after multiple attempts."

# Define the agents
def create_outline_agent():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLAN_PROMPT),
        ("human", "{topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def outline_agent(state: AgentState) -> AgentState:
        outline = chain.invoke({"topic": state["topic"]})
        return {**state, "outline": outline}
    
    return outline_agent

def create_research_agent():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCH_PLAN_PROMPT),
        ("human", "{topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def research_agent(state: AgentState) -> AgentState:
        # Generate research queries
        queries = chain.invoke({"topic": state["topic"]})
        queries = [q.strip() for q in queries.split('\n') if q.strip()][:3]
        
        # Perform research with retry logic
        research_results = []
        for query in queries:
            result = search_with_retry(query)
            research_results.append(f"Query: {query}\nResults:\n{result}\n")
        
        research_content = "\n".join(research_results)
        return {**state, "research": research_content}
    
    return research_agent

def create_writer_agent():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_PROMPT),
        ("human", "Topic: {topic}\nOutline: {outline}\nResearch: {research}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def writer_agent(state: AgentState) -> AgentState:
        essay = chain.invoke({
            "topic": state["topic"],
            "outline": state["outline"],
            "research": state["research"]
        })
        return {**state, "essay": essay}
    
    return writer_agent

def create_critique_agent():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", REFLECTION_PROMPT),
        ("human", "{essay}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def critique_agent(state: AgentState) -> AgentState:
        critique = chain.invoke({"essay": state["essay"]})
        return {**state, "critique": critique}
    
    return critique_agent

def create_research_critique_agent():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCH_CRITIQUE_PROMPT),
        ("human", "Essay: {essay}\nCritique: {critique}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def research_critique_agent(state: AgentState) -> AgentState:
        # Generate research queries based on critique
        queries = chain.invoke({
            "essay": state["essay"],
            "critique": state["critique"]
        })
        queries = [q.strip() for q in queries.split('\n') if q.strip()][:3]
        
        # Perform research with retry logic
        research_results = []
        for query in queries:
            result = search_with_retry(query)
            research_results.append(f"Query: {query}\nResults:\n{result}\n")
        
        additional_research = "\n".join(research_results)
        # Append new research to existing research
        updated_research = state["research"] + "\n\nAdditional Research Based on Critique:\n" + additional_research
        return {**state, "research": updated_research}
    
    return research_critique_agent

def create_revision_agent():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_PROMPT),
        ("human", "Previous Essay: {essay}\nCritique: {critique}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def revision_agent(state: AgentState) -> AgentState:
        revised_essay = chain.invoke({
            "essay": state["essay"],
            "critique": state["critique"]
        })
        return {**state, "essay": revised_essay, "iteration": state["iteration"] + 1}
    
    return revision_agent

def should_continue(state: AgentState) -> str:
    # Check if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    
    # Check if word count is within limit
    current_word_count = len(state["essay"].split())
    if current_word_count > state["word_count_limit"]:
        return "end"
    
    return "continue"

# Create the workflow
def create_essay_workflow():
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes with unique names that don't conflict with state keys
    workflow.add_node("create_outline", create_outline_agent())
    workflow.add_node("perform_research", create_research_agent())
    workflow.add_node("write_essay", create_writer_agent())
    workflow.add_node("provide_critique", create_critique_agent())
    workflow.add_node("research_critique", create_research_critique_agent())
    workflow.add_node("revise_essay", create_revision_agent())
    
    # Add edges with updated node names
    workflow.add_edge("create_outline", "perform_research")
    workflow.add_edge("perform_research", "write_essay")
    workflow.add_edge("write_essay", "provide_critique")
    workflow.add_edge("provide_critique", "research_critique")
    workflow.add_edge("research_critique", "revise_essay")
    
    # Add conditional edges for revision cycle
    workflow.add_conditional_edges(
        "revise_essay",
        should_continue,
        {
            "continue": "provide_critique",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("create_outline")
    
    return workflow.compile()

def process_essay_request(topic: str, max_iterations: int = 3, word_count_limit: int = 1000) -> Dict[str, Any]:
    print(f"\nStarting essay research for topic: {topic}")
    
    # Validate parameters
    max_iterations = min(max(1, max_iterations), 10)  # Cap between 1 and 10
    word_count_limit = max(100, word_count_limit)  # Minimum 100 words
    
    # Initialize the workflow
    workflow = create_essay_workflow()
    
    # Initialize the state
    initial_state = {
        "messages": [],
        "topic": topic,
        "outline": "",
        "research": "",
        "essay": "",
        "critique": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "word_count_limit": word_count_limit
    }
    
    print(f"\nInitializing workflow with max iterations: {max_iterations} and word count limit: {word_count_limit}")
    
    # Run the workflow
    final_state = workflow.invoke(initial_state)
    
    print("\nWorkflow completed!")
    
    return {
        "topic": final_state["topic"],
        "outline": final_state["outline"],
        "final_essay": final_state["essay"],
        "final_critique": final_state["critique"],
        "iterations": final_state["iteration"],
        "word_count": len(final_state["essay"].split())
    }

def main():
    st.set_page_config(page_title="Essay Research Assistant", page_icon="📚", layout="wide")
    
    st.title("📚 Essay Research Assistant")
    st.markdown("""
    This tool helps you research and write essays by:
    1. Creating an outline
    2. Conducting research
    3. Writing the essay
    4. Providing critique
    5. Revising the essay
    """)
    
    # Get API keys from environment variables
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable")
        return
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Please set the GOOGLE_API_KEY environment variable")
        return
    if not os.getenv("GOOGLE_CSE_ID"):
        st.error("Please set the GOOGLE_CSE_ID environment variable")
        return
    
    # Input for topic
    topic = st.text_input("Enter your research topic:", 
                         placeholder="e.g., The Benefits of Regular Exercise")
    
    # Input for max iterations
    max_iterations = st.slider("Maximum number of revisions:", 
                             min_value=1, 
                             max_value=10, 
                             value=3,
                             help="Maximum number of times the essay will be revised (1-10)")
    
    # Input for word count limit
    word_count_limit = st.number_input("Maximum word count:", 
                                     min_value=100, 
                                     max_value=5000, 
                                     value=1000,
                                     step=100,
                                     help="Maximum number of words in the final essay")
    
    if st.button("Generate Essay"):
        if not topic:
            st.warning("Please enter a topic first!")
            return
            
        with st.spinner("Generating your essay... This may take a few minutes."):
            try:
                result = process_essay_request(topic, max_iterations, word_count_limit)
                
                # Display results in expandable sections
                with st.expander("📝 Outline", expanded=True):
                    st.markdown(result['outline'])
                
                with st.expander("📄 Final Essay", expanded=True):
                    st.markdown(result['final_essay'])
                    st.info(f"Word count: {result['word_count']}")
                
                with st.expander("📋 Final Critique", expanded=True):
                    st.markdown(result['final_critique'])
                
                st.info(f"Number of iterations completed: {result['iterations']}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Stack trace:")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

