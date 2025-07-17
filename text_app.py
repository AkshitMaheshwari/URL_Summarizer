import os
import streamlit as st
import validators
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

load_dotenv()

st.title("Yt video and Webpages summary")
st.subheader("Link daal")

groq_api_key =os.getenv("GROQ_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(model="GPT-4.1", base_url = "https://models.inference.ai.azure.com",api_key = api_key)
llm = ChatGroq(model = "Gemma2-9b-It",groq_api_key = groq_api_key)

prompt_template = """
You are a helpful assistant. Provide the title, detailed explanation and conclusion of the content:
Content:{text}
Summary:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
final_prompt = """
Provide the final summary of the entire content with these important points.
Add a proper heading and ensure that the summary is properly written with detailed description as well as conclusions and one can easily understand the content and if it is in different language then translate it to english
Content:{text}
"""
final_prompt_tem = PromptTemplate(template=final_prompt, input_variables=["text"])

url = st.text_input("Enter URL here")

from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

def get_transcript_from_url(youtube_url):
    try:
        parsed_url = urlparse(youtube_url)
        video_id = ""

        if "youtube.com" in youtube_url:
            video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        elif "youtu.be" in youtube_url:
            video_id = parsed_url.path.lstrip("/")
        else:
            raise ValueError("URL does not seem to be a valid YouTube link.")

        if not video_id:
            raise ValueError("Could not extract video ID from URL.")

        # Try both languages
        transcript = None
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
        except NoTranscriptFound:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            except NoTranscriptFound:
                raise RuntimeError("No transcript found in Hindi or English.")
        except TranscriptsDisabled:
            raise RuntimeError("Transcripts are disabled for this video.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while fetching transcript: {str(e)}")

        if not transcript:
            raise RuntimeError("Transcript data is empty or could not be fetched.")

        text = " ".join([t["text"] for t in transcript])
        return text

    except Exception as e:
        raise RuntimeError(f"Failed to get transcript: {str(e)}")



if st.button("Summarize Content"):
    if not url.strip():
        st.error("Please enter a URL.")
    elif not validators.url(url):
        st.error("Invalid URL format.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):

                if "youtube.com" in url or "youtu.be" in url:
                    transcript_text = get_transcript_from_url(url)
                    docs = [Document(page_content=transcript_text)]

                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                st.write("🔍 Preview of content:", docs[0].page_content[:300] + "...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap = 100)
                split_docs = text_splitter.split_documents(docs)

                chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt,combine_prompt=final_prompt_tem)
                summary = chain.run(split_docs)

                st.success("Summary:")
                st.write(summary)

        except Exception as e:
            st.error("An error occurred while processing the URL.")
            st.exception(e)
