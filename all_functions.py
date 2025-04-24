import os

import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from mistralai import Mistral
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import whisper
from pydub import AudioSegment
import tempfile

# Logging setup
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Initialize the Mistral client
api_key = os.getenv("MISTRAL_KEY")
model = "pixtral-12b-2409"
client = Mistral(api_key=api_key)

# Initialize the SentenceTransformer for embedding
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Read the PDF content
pdf_path = "./Piaic.pdf"
reader = PdfReader(pdf_path)

# Extract text from each page of the PDF
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text.strip():  # Avoid empty pages
        documents.append({"id": i, "content": text})

# Initialize Qdrant in memory
qdrant = QdrantClient(":memory:")

# Create a collection in Qdrant for storing the embeddings
qdrant.create_collection(
    collection_name="pdf_content",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)


qdrant.upload_points(
    collection_name="pdf_content",
    points=[
        models.PointStruct(
            id=doc["id"],
            vector=encoder.encode(doc["content"]).tolist(),
            payload={"content": doc["content"]},
        )
        for doc in documents
    ],
)

# Function to search for relevant content
def search_documents(query):
    hits = qdrant.search(
        collection_name="pdf_content",
        query_vector=encoder.encode(query).tolist(),
        limit=2,  # Return top 2 results
        score_threshold=0.2,  # Define a relevance threshold
    )
    return [hit.payload["content"] for hit in hits]




# Flask setup
app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load Whisper model once at startup
whisper_model = whisper.load_model("base")

message_history : list[] = []

@app.route("/chat", methods=["POST"])
async def chat():
    global message_history
    # Get user input and image URL from the request
    data = request.get_json()
    user_input = request.json.get("message", "")
    image_urls = request.json.get("image_urls", [])

    if not user_input and not image_urls:
        return jsonify({"error": "No message or image URLs provided"}), 400
    
    # Combine text and images into content
    content = []
    if user_input:
        content.append({"type": "text", "text": user_input})
    for url in image_urls[:2]:  # Limit to 2 images
        content.append({"type": "image_url", "image_url": str(url)})

    messages = message_history + [{"role": "user", "content": content}]
    

   
    if user_input and documents:
        hits = search_documents(user_input)
        context = "\n".join(hits)
        system_messages= {
            "role": "system",
            "content": f"Here is knowledge:\n{context}\n----------------------\n Use this knowledge ONLY IF IT IS RELEVANT to answer the user's query. If the user asks something irrelevant (outside the document), reply normally and do not mention the document. Give answer in one line and dont confuse the user by multiple answers. Talk in friendly way"
        }
        messages = [system_messages] + messages
    else:
        messages = [{"role": "system", "content": "Answer in friendly, single-line way."}] + messages
        
    
    try:
    # Generate a response using Mistral AI
        chat_response = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens = 300
        )

    # Extract and return the response
        response_content = chat_response.choices[0].message.content
        logging.info(f"Response: {response_content}")

        message_history.append({"role": "user", "content": content})
        message_history.append({"role": "assistant", "content": response_content})

        return jsonify({"response": response_content})
    except Exception as e:
        logging.error(f"Error in chat completion: {e}")
        return jsonify({"error": "Failed to process the request"}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handle audio transcription requests using Whisper."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    try:
        # Save uploaded file
        audio_file = request.files['audio']
        webm_path = tempfile.mktemp(suffix='.webm')
        wav_path = tempfile.mktemp(suffix='.wav')
        audio_file.save(webm_path)

        # Convert to WAV format
        audio = AudioSegment.from_file(webm_path, format="webm")  # Read file using ffmpeg
        audio.export(wav_path, format="wav")  # Export file to WAV format

        # Transcribe audio
        result = whisper_model.transcribe(wav_path)
        transcribed_text = result['text']

        # Cleanup temporary files
        os.remove(webm_path)
        os.remove(wav_path)

        return jsonify({'text': transcribed_text})

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)



