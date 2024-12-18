import time
from flask import Flask, request, render_template, make_response, session
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube, extract
import textwrap
import openai
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from google.cloud import speech
import io
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urlparse
import traceback

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Add secret key for sessions

transcript = ""
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)


class WebMetadataScraper:
    def __init__(self, url):
        """
        Initialize the web scraper with a given URL
        
        Args:
            url (str): The website URL to scrape
        """
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.metadata = {
            'url': url,
            'source_methods': []
        }

    def fetch_html(self):
        """
        Fetch the HTML content of the webpage
        
        Returns:
            str: HTML content or None if failed
        """
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching HTML: {e}")
            return None

    def extract_metadata_from_html(self, html):
        """
        Extract metadata using various HTML parsing techniques
        
        Args:
            html (str): HTML content of the webpage
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Extract title
        self._extract_title(soup)
        
        # Extract description
        self._extract_description(soup)
        
        # Extract author/owner
        self._extract_author(soup)
        
        # Extract keywords
        self._extract_keywords(soup)
        
        # Extract structured data
        self._extract_structured_data(soup)

    def _extract_title(self, soup):
        """Extract page title from different sources"""
        # Method 1: HTML title tag
        if soup.title and soup.title.string:
            self.metadata['title'] = soup.title.string.strip()
            self.metadata['source_methods'].append('html_title')

        # Method 2: Open Graph title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            self.metadata['og_title'] = og_title['content'].strip()
            self.metadata['source_methods'].append('og_title')

    def _extract_description(self, soup):
        """Extract page description from different sources"""
        # Method 1: Meta description tag
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            self.metadata['description'] = meta_desc['content'].strip()
            self.metadata['source_methods'].append('meta_description')

        # Method 2: Open Graph description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            self.metadata['og_description'] = og_desc['content'].strip()
            self.metadata['source_methods'].append('og_description')

    def _extract_author(self, soup):
        """Extract author/owner information"""
        # Method 1: Meta author tag
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            self.metadata['author'] = meta_author['content'].strip()
            self.metadata['source_methods'].append('meta_author')
            print(f"Author: {meta_author['content']}")

        # Method 2: Specific HTML elements often used for author
        author_elements = [
            soup.find(class_=re.compile(r'author', re.IGNORECASE)),
            soup.find(id=re.compile(r'author', re.IGNORECASE))
        ]
        for element in author_elements:
            if element and element.get_text(strip=True):
                self.metadata['page_author'] = element.get_text(strip=True)
                self.metadata['source_methods'].append('page_author_element')
                break

    def _extract_keywords(self, soup):
        """Extract keywords"""
        # Method 1: Meta keywords tag
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            self.metadata['keywords'] = [
                kw.strip() for kw in meta_keywords['content'].split(',')
            ]
            self.metadata['source_methods'].append('meta_keywords')

    def _extract_structured_data(self, soup):
        """
        Extract structured data from JSON-LD and other sources
        """
        # Extract JSON-LD structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        structured_data = []
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                structured_data.append(data)
                
                # Extract additional metadata from structured data
                if isinstance(data, dict):
                    if '@type' in data:
                        self.metadata['structured_type'] = data['@type']
                    if 'name' in data:
                        self.metadata['structured_name'] = data['name']
            except json.JSONDecodeError:
                pass
        
        if structured_data:
            self.metadata['structured_data'] = structured_data
            self.metadata['source_methods'].append('json_ld')

    def scrape(self):
        """
        Main method to scrape website metadata
        
        Returns:
            dict: Extracted metadata
        """
        try:
            # Fetch HTML content
            html = self.fetch_html()
            
            if not html:
                return self.metadata
            
            # Extract metadata from HTML
            self.extract_metadata_from_html(html)
            
            return self.metadata
        
        except Exception as e:
            print(f"Error in web scraping: {e}")
            traceback.print_exc()
            return self.metadata
        

# Set your OpenAI API key as an environment variable for security
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

import os
from googleapiclient.discovery import build
def get_video_metadata(video_id):
    """
    Retrieve metadata for a YouTube video using YouTube Data API v3.
    
    Args:
        video_id (str): The ID of the YouTube video
        api_key (str): Your YouTube Data API key
    
    Returns:
        dict: A dictionary containing video metadata
    """
    try:
        # Build the YouTube API client
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Request video details
        video_response = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=video_id
        ).execute()
        
        # Check if video exists
        if not video_response['items']:
            print("No video found with the given ID.")
            return None
        
        # Extract video details
        video = video_response['items'][0]
        snippet = video['snippet']
        statistics = video.get('statistics', {})
        
        # Prepare metadata dictionary
        # metadata = {
        #     'title': snippet.get('title', 'N/A'),
        #     'description': snippet.get('description', 'N/A'),
        #     'channel_title': snippet.get('channelTitle', 'N/A'),
        #     'channel_id': snippet.get('channelId', 'N/A'),
        #     'published_at': snippet.get('publishedAt', 'N/A'),
        #     'tags': snippet.get('tags', []),
        #     'category_id': snippet.get('categoryId', 'N/A'),
        #     'view_count': statistics.get('viewCount', '0'),
        #     'like_count': statistics.get('likeCount', '0'),
        #     'comment_count': statistics.get('commentCount', '0')
        # }
        # metadata = {
        #     'title': yt.title,
        #     'description': yt.description,
        #     'channel_name': yt.author,
        #     'channel_id': yt.channel_id,
        #     'publish_date': str(yt.publish_date),
        #     'views': yt.views,
        #     'length': yt.length,
        #     'thumbnail_url': yt.thumbnail_url,
        # }
        
        # return metadata
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def get_video_id(url):
    try:
        return extract.video_id(url)
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        return None

def download_audio(video_url, output_path="audio.mp3"):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output_path)
    return output_path


def transcribe_audio(file_path):
    client = speech.SpeechClient()

    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcription = []
    for result in response.results:
        transcription.append(result.alternatives[0].transcript)

    return transcription

def fetch_transcript(video_id, video_url):
    """
    Fetch the transcript for a video.
    If English transcription is not available, fetch the auto-generated transcription and translate it into English.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # First try to get any English transcript (manual or auto-generated)
        try:
            # Try getting any English transcript
            transcript = transcript_list.find_transcript(['en'])
            return transcript.fetch()
        except NoTranscriptFound:
            # If no English transcript, try getting any available transcript
            try:
                available_transcripts = transcript_list.find_generated_transcript(['en', 'ja', 'ko', 'es', 'fr', 'de', 'hi'])
                if available_transcripts:
                    non_english_transcript = available_transcripts.fetch()
                    logger.info("Found non-English transcript. Attempting translation.")
                    return translate_transcript_to_english(non_english_transcript)
            except Exception as e:
                logger.error(f"Error fetching available transcripts: {str(e)}")
                return None
                
    except TranscriptsDisabled:
        logger.error("Transcripts are disabled for this video.")
        try:
            output_path = download_audio(video_url)
            return transcribe_audio(output_path)
        except Exception as e:
            logger.info(f"Audio Error: {str(e)}")
            return None
    except NoTranscriptFound:
        logger.error("No transcripts found for this video.")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return None


def translate_transcript_to_english(transcript):
    """
    Translate a transcript to English using an AI model.
    """
    try:
        # Combine transcript text into one string
        full_text = ' '.join([entry['text'] for entry in transcript])
        
        # Use AI model to translate
        prompt = f"""
        Translate the following transcript into English:
        {full_text}
        """
        
        response = gemini.generate_content(prompt)
        translated_text = response.text.strip()
        
        # Return translated transcript in the original format
        return [{"text": line} for line in translated_text.split('\n') if line.strip()]
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return [{"text": "Translation failed. Original transcript: " + ' '.join([entry['text'] for entry in transcript])}]


def translate_text(text):
    """
    Translate text to English using the AI model.
    """
    try:
        prompt = f"""
        Translate the following text into English:
        {text}
        """
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails


def summarize_text(text):
    """
    Summarize text using the AI model.
    """
    try:
        system_instructions = """You are a skilled content creator and assistant specializing in summarizing long-form content and transforming it into engaging, reader-friendly blogs. Your task is to analyze the provided YouTube video transcript, extract key points, and craft a well-structured blog post that captivates readers.
            Guidelines for the Blog:

            Summary: Begin with a concise introduction summarizing the video's main topic and purpose. Highlight why it’s relevant or valuable to the reader.
            Key Points: Break down the transcript into logical sections, extracting essential ideas, insights, and takeaways. Use bullet points or subheadings for clarity where appropriate.
            Tone: Maintain an informative, engaging, and conversational tone throughout the blog. Adjust the tone based on the video's subject (e.g., professional for a tech tutorial, lighthearted for lifestyle content).
            Actionable Insights: Provide actionable tips or advice wherever applicable, helping readers apply the video’s content to their own lives or work.
            Conclusion: End with a strong conclusion summarizing the key messages and encouraging readers to engage further (e.g., watch the video, comment, share).
            SEO Optimization: Suggest a compelling title and meta description for the blog to optimize it for search engines.
            Deliverable Format:

            Provide the blog in markdown format with appropriate headings, subheadings, and formatting.
            Include a list of keywords for SEO and suggest a title that grabs attention.
            Identify the speaker and display the speaker name in the blog header.
            Transcript:"""
        text = system_instructions + "\n" + text
        total_tokens = gemini.count_tokens(text).total_tokens
        logger.info(f"Gemini Token count: {total_tokens}")
        response = gemini.generate_content(text)
        # logger.info(f"Response: {response.text}")
        return response.text
    except Exception as e:
        logger.info(f"Exception in Summarize text: {str(e)}")
        return "An error occurred during summarization."


# def generate_summary(transcript):
#     """
#     Generate a summary from the transcript using AI.
#     """
#     full_text = ' '.join([entry['text'] for entry in transcript])
#     chunks = split_text(full_text)
#     summaries = [summarize_text(chunk) for chunk in chunks]
#     return ' '.join(summaries)


@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    transcript_text = ""
    if request.method == 'POST':
        global transcript
        logger.info("Loading...")
        video_url = request.form['url']
        video_id = get_video_id(video_url)
        logger.info(f"Attempting to fetch transcript for video ID: {video_id}")
        
        if video_id:
            transcript = fetch_transcript(video_id, video_url)
            if transcript:
                transcript_text = ' '.join([entry['text'] for entry in transcript])
            else:
                logger.error(f"Failed to fetch transcript for video ID: {video_id}")
                summary = "No transcripts available for this video. Please try another video."
                return render_template('index.html', summary=summary)
            
            print(f"\nScraping metadata for: {video_url}")
            scraper = WebMetadataScraper(video_url)
            metadata = scraper.scrape()
            
            # Pretty print metadata
            for key, value in metadata.items():
                print(f"{key}: {value}")

            if metadata is None:
                logger.info(f"Failed to fetch metadata for video URL: {video_url}")
            summary = summarize_text(f"Metadata: {metadata}\n\nTranscript: {transcript_text}")
            # logger.info(f"Transcript summarized successfully {summary}")
            return render_template('index.html', summary=summary, metadata=metadata)
        else:
            summary = "Invalid YouTube URL."

    return render_template('index.html', summary=summary)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return {"response": "Please provide a valid question."}, 400

        # Use the transcript from the global variable
        global transcript
        context = transcript
        logger.info(f"Transcript : {len(context)}")
        if not context:
            return {"response": "Transcript not loaded. Please summarize a video first."}, 400

        # Prepare the prompt
        prompt = f"""
        You are a highly intelligent assistant with knowledge of the following video details:
        {context}
        
        The user has asked: "{question}". Please provide a clear and concise response.
        """

        # Generate a response using the AI model
        response = gemini.generate_content(prompt).text

        return {"response": response}, 200

    except Exception as e:
        logger.exception(f"Error during chat: {str(e)}")
        return {"response": "An error occurred while processing your request."}, 500


if __name__ == '__main__':
    # Ensure the OpenAI API key is set
    if not openai.api_key:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    # Run the Flask app
    app.run(debug=True)
