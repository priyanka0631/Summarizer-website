import os
from flask import Flask, render_template, request, jsonify, make_response
from bs4 import BeautifulSoup
import requests
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
import pytesseract
from PIL import Image
import cv2
import numpy as np
import whisper
from pydub import AudioSegment
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

pegasus_model_name = "google/pegasus-xsum"
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
# Load Whisper ASR model
asr_model = whisper.load_model("base")

# Load mBART summarization pipeline for multilingual text summarization
mbart_summarizer = pipeline("summarization", model="facebook/mbart-large-50")

# Function to create bullet points from the text summary
def generate_bullet_points(text):
    sentences = text.split('.')
    bullet_points = "<ul>"
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            bullet_points += f"<li>{sentence}</li>"
    bullet_points += "</ul>"
    return bullet_points

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/text-summarization", methods=["POST"])
def text_summarization():
    input_text = request.form.get("inputtext_")
    summary_length = request.form.get("summary_length", "medium")
    
    if input_text:
        try:
            # Detect the language of the input text
            detected_language = detect(input_text)
        except LangDetectException:
            detected_language = "unknown"
        
        # Use mBART summarizer for multilingual text
        try:
            if summary_length == "short":
                summary = mbart_summarizer(input_text, max_length=20, min_length=10, do_sample=False)
            elif summary_length == "long":
                summary = mbart_summarizer(input_text, max_length=100, min_length=50, do_sample=False)
            else:  # Default (medium) summary
                summary = mbart_summarizer(input_text, max_length=50, min_length=25, do_sample=False)
            
            decoded_summary = summary[0]['summary_text']
            bullet_points_html = generate_bullet_points(decoded_summary)
            
            return render_template(
                "index.html",
                summary=decoded_summary,
                bullet_points_html=bullet_points_html,
                summary_length=summary_length,
                detected_language=detected_language,
            )
        except Exception as e:
            return render_template("index.html", summary=f"Error during summarization: {str(e)}", detected_language=detected_language)
    
    return render_template("index.html", summary="Error: No input text provided.")

@app.route("/url-summarization", methods=["POST"])
def url_summarization():
    url = request.form.get("url")
    
    if not url:
        return render_template("index.html", summary="Error: No URL provided.")
    
    try:
        # Request the URL content
        response = requests.get(url)
        response.raise_for_status()

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ')

        # Tokenize and summarize the content using Pegasus
        tokens = pegasus_tokenizer(text_content, truncation=True, padding="longest", return_tensors="pt")
        summary = pegasus_model.generate(tokens['input_ids'], max_length=50, min_length=25, length_penalty=1.5, early_stopping=True)
        decoded_summary = pegasus_tokenizer.decode(summary[0], skip_special_tokens=True)
        
        # Generate bullet points for the URL summary
        bullet_points_html = generate_bullet_points(decoded_summary)
        
        return render_template("index.html", summary=decoded_summary, bullet_points_html=bullet_points_html)
    except Exception as e:
        return render_template("index.html", summary=f"Error: {str(e)}")

@app.route("/download-summary", methods=["POST"])
def download_summary():
    summary_content = request.form.get("summary_content")
    if summary_content:
        response = make_response(summary_content)
        response.headers["Content-Disposition"] = "attachment; filename=summary.txt"
        response.headers["Content-Type"] = "text/plain"
        return response
    return "Error: No summary content to download.", 400

@app.route("/audio-summarization", methods=["POST"])
def audio_summarization():
    if 'audio_file' not in request.files:
        return render_template("index.html", summary="No audio file provided.")

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return render_template("index.html", summary="No selected file.")

    try:
        # Save the uploaded audio file
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        audio_path = os.path.join(upload_folder, audio_file.filename)
        audio_file.save(audio_path)

        # Transcribe the audio file
        transcription = asr_model.transcribe(audio_path)["text"]

        # Summarize the transcribed text using Pegasus
        tokens = pegasus_tokenizer(transcription, truncation=True, padding="longest", return_tensors="pt")
        summary = pegasus_model.generate(tokens['input_ids'], max_length=50, min_length=25, length_penalty=1.5, early_stopping=True)
        decoded_summary = pegasus_tokenizer.decode(summary[0], skip_special_tokens=True)

        # Generate bullet points
        bullet_points_html = generate_bullet_points(decoded_summary)

        return render_template("index.html", summary=decoded_summary, bullet_points_html=bullet_points_html)
    except Exception as e:
        return render_template("index.html", summary=f"Error processing audio: {str(e)}")

@app.route("/image-summarization", methods=["POST"])
def image_summarization():
    if 'image_file' not in request.files:
        return render_template("index.html", summary="No image file provided.")

    image_file = request.files['image_file']

    if image_file.filename == '':
        return render_template("index.html", summary="No selected file.")

    try:
        # Save the uploaded image
        upload_folder = 'static/uploads/images'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        image_path = os.path.join(upload_folder, image_file.filename)
        image_file.save(image_path)

        # Extract text from the image
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return render_template("index.html", summary="No text found in the image.")

        # Summarize the extracted text using Pegasus
        tokens = pegasus_tokenizer(extracted_text, truncation=True, padding="longest", return_tensors="pt")
        summary = pegasus_model.generate(tokens['input_ids'], max_length=50, min_length=25, length_penalty=1.5, early_stopping=True)
        decoded_summary = pegasus_tokenizer.decode(summary[0], skip_special_tokens=True)

        # Generate bullet points
        bullet_points_html = generate_bullet_points(decoded_summary)

        return render_template("index.html", summary=decoded_summary, bullet_points_html=bullet_points_html)
    except Exception as e:
        return render_template("index.html", summary=f"Error processing image: {str(e)}")

@app.route("/chart-summarization", methods=["POST"])
def chart_summarization():
    if 'chart_file' not in request.files:
        return render_template("index.html", summary="No chart file provided.")

    chart_file = request.files['chart_file']

    if chart_file.filename == '':
        return render_template("index.html", summary="No selected file.")

    try:
        # Save the uploaded chart
        upload_folder = 'static/uploads/charts'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        chart_path = os.path.join(upload_folder, chart_file.filename)
        chart_file.save(chart_path)

        # Process the chart using OpenCV (or similar library)
        image = cv2.imread(chart_path)
        # (Placeholder for text/numerical data extraction logic)

        # Example: Use pytesseract for text extraction
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return render_template("index.html", summary="No text found in the chart.")

        # Summarize the extracted text using Pegasus
        tokens = pegasus_tokenizer(extracted_text, truncation=True, padding="longest", return_tensors="pt")
        summary = pegasus_model.generate(tokens['input_ids'], max_length=50, min_length=25, length_penalty=1.5, early_stopping=True)
        decoded_summary = pegasus_tokenizer.decode(summary[0], skip_special_tokens=True)

        # Generate bullet points
        bullet_points_html = generate_bullet_points(decoded_summary)

        return render_template("index.html", summary=decoded_summary, bullet_points_html=bullet_points_html)
    except Exception as e:
        return render_template("index.html", summary=f"Error processing chart: {str(e)}")

if __name__ == "__main__":
     # Initialize the Pegasus model and tokenizer
    
    app.run(host='0.0.0.0', port=5000, debug=True)
