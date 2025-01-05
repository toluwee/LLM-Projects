import os
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# Initialize OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
llm = ChatOllama(model = "gemma:2b")

def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    try:
        parsed_url = urlparse(youtube_url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            if parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            if parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]
    except Exception:
        return None
    return None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """Fetch transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        return None


# First prompt to extract all key points
extract_points_template = """
You are an expert content analyst. Your task is to thoroughly analyze the provided YouTube transcript and extract ALL key points, insights, examples, and arguments made in the video. 

Process:
1. First, read through the entire transcript carefully
2. Create a comprehensive list of EVERY distinct point, insight, or example mentioned
3. Include even seemingly minor points - we want to capture everything
4. Number each point sequentially
5. Keep the points in the same order as they appear in the video
6. For each point, include:
   - The main idea
   - Any supporting details or examples given
   - Any relevant context or connections made

Transcript: {transcript}

Please list ALL points made in the video, ensuring nothing is missed.
"""

points_prompt = PromptTemplate(
    input_variables=["transcript"],
    template=extract_points_template
)

# Second prompt to create the SEO article using the extracted points
article_template = """
Act as an expert copywriter specializing in content optimization for SEO. Your task is to transform the provided YouTube transcript points into a comprehensive, well-structured article. 

Here are the extracted key points from the video:
{extracted_points}

Your objectives:

Content Organization:
1. Create a logical structure that incorporates EVERY point provided above
2. Ensure no information or examples from the original points are lost
3. Group related points into coherent sections

Content Development:
1. Expand on each point while maintaining accuracy
2. Include all examples and context from the original points
3. Add appropriate transitions between points
4. Maintain the depth and nuance of the original content

SEO Optimization:
1. Identify and incorporate primary and secondary keywords naturally
2. Create SEO-optimized headings that reflect the content structure
3. Maintain proper heading hierarchy (H1, H2, H3)
4. Include a compelling meta title and description

Writing Style:
1. Make the content engaging and reader-friendly
2. Use clear, professional language
3. Maintain proper flow and transitions
4. Avoid repetition while ensuring completeness

Please provide:
1. Suggested Meta Title (60-65 characters)
2. Suggested Meta Description (150-160 characters)
3. The full article with proper heading structure, incorporating ALL points from the original list

Remember: Every point from the extracted list must be included in the final article - nothing should be left out.
"""

article_prompt = PromptTemplate(
    input_variables=["extracted_points"],
    template=article_template
)


def process_transcript(transcript: str) -> Optional[str]:
    """Process the YouTube transcript using a two-step approach"""
    try:
        # Step 1: Extract all points
        points_response = llm.invoke(points_prompt.format(transcript=transcript))
        extracted_points = points_response.content

        # Step 2: Create article from extracted points
        article_response = llm.invoke(article_prompt.format(extracted_points=extracted_points))

        # Return both the points and the article
        return {
            "extracted_points": extracted_points,
            "article": article_response.content
        }
    except Exception as e:
        return f"Error processing transcript: {str(e)}"


# Streamlit UI
st.title("YouTube Video to Comprehensive SEO Article Converter")
st.write("Enter a YouTube video URL to convert its content into a detailed SEO-optimized article")

youtube_url = st.text_input("YouTube Video URL")

if st.button("Convert to Article"):
    if youtube_url:
        with st.spinner("Processing..."):
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check the URL and try again.")
            else:
                transcript = get_youtube_transcript(video_id)
                if not transcript:
                    st.error("Could not fetch transcript. Make sure the video has closed captions available.")
                else:
                    result = process_transcript(transcript)
                    if isinstance(result, dict):
                        st.subheader("Extracted Key Points")
                        st.markdown(result["extracted_points"])
                        st.subheader("Generated Article")
                        st.markdown(result["article"])
                    else:
                        st.error(result)
    else:
        st.warning("Please enter a YouTube video URL")

# Add instructions and supported formats
st.markdown("---")
st.markdown("""
### How to Use
1. Paste the YouTube video URL
2. Click 'Convert to Article'
3. Review both the extracted points and the final article
4. The output includes:
   - Complete list of all points from the video
   - SEO-optimized article incorporating all points
   - Meta title and description
""")

st.markdown("""
### Supported URL Formats:
- https://www.youtube.com/watch?v=VIDEO_ID
- https://youtu.be/VIDEO_ID
- https://www.youtube.com/embed/VIDEO_ID
""")