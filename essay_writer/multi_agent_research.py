import os
from typing import List, Dict, Any, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
import json
import time

# Existing prompts
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
    if state["iteration"] >= 3:  # Maximum 3 iterations
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
    workflow.add_node("revise_essay", create_revision_agent())
    
    # Add edges with updated node names
    workflow.add_edge("create_outline", "perform_research")
    workflow.add_edge("perform_research", "write_essay")
    workflow.add_edge("write_essay", "provide_critique")
    workflow.add_edge("provide_critique", "revise_essay")
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

def process_essay_request(topic: str) -> Dict[str, Any]:
    print(f"\nStarting essay research for topic: {topic}")
    
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
        "iteration": 0
    }
    
    print("\nInitializing workflow...")
    
    # Run the workflow
    final_state = workflow.invoke(initial_state)
    
    print("\nWorkflow completed!")
    
    return {
        "topic": final_state["topic"],
        "outline": final_state["outline"],
        "final_essay": final_state["essay"],
        "final_critique": final_state["critique"],
        "iterations": final_state["iteration"]
    }

def main():
    # Get API keys from environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Please set the GOOGLE_API_KEY environment variable")
    if not os.getenv("GOOGLE_CSE_ID"):
        raise ValueError("Please set the GOOGLE_CSE_ID environment variable")
    
    # Test with a simpler topic
    test_topics = [
        "The Benefits of Regular Exercise",
        "The Importance of Reading Books",
        "The Role of Technology in Education"
    ]
    
    # Test with the first topic
    topic = test_topics[0]
    print(f"\nTesting with topic: {topic}")
    
    try:
        result = process_essay_request(topic)
        
        print("\n=== Results ===")
        print(f"\nTopic: {result['topic']}")
        print("\nOutline:")
        print(result['outline'])
        print("\nFinal Essay:")
        print(result['final_essay'])
        print("\nFinal Critique:")
        print(result['final_critique'])
        print(f"\nNumber of iterations: {result['iterations']}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

