import os
import json
import python_weather
import asyncio
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

# Weather function
async def retreive_get_weather(city: str) -> str:
    async with python_weather.Client(unit=python_weather.METRIC) as client:
        weather = await client.get(city)
        temperatures = weather.temperature
        print(f"Current temperature in {city}: {temperatures}")
        return json.dumps({'temperature': temperatures})

# Tools for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "retreive_get_weather",
            "description": "Get current weather of any city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

names_to_functions = {
    "retreive_get_weather": retreive_get_weather
}

# Flask setup
app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load Whisper model once at startup
whisper_model = whisper.load_model("base")

message_history = []


@app.route("/chat", methods=["POST"])
async def chat():
    global message_history
    data = request.get_json()
    user_input = data.get("message", "")
    image_urls = data.get("image_urls", [])

    if not user_input and not image_urls:
        return jsonify({"error": "No message or image URLs provided"}), 400

    # Initialize messages list
    messages = []

    # Handle weather-related queries
    if any(keyword in user_input.lower() for keyword in ["temperature", "weather"]):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide weather information. Use the tool's output to respond to the user's query in a friendly, single-line message."},
            {"role": "user", "content": user_input}
        ]
        chat_response = client.chat.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="any"
        )

        messages.append(chat_response.choices[0].message)
        
        tool_call = chat_response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)
        
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        function_result = await names_to_functions[function_name](**function_params)
        
        messages.append({
            "role": "tool",
            "name": function_name,
            "content": f"use this knowledge {function_result} in order to respond to the user query. Activate function calling and give a single-line response. Talk in a friendly way.",
            "tool_call_id": tool_call.id
        })
        
        chat_response = client.chat.complete(
            model=model,
            messages=messages,
        )

        response_content = chat_response.choices[0].message.content
        logging.info(f"Weather response: {response_content}")
        return jsonify({"response": response_content})

    # Handle PDF search for non-weather queries
    if user_input and documents:
        hits = search_documents(user_input)
        context = "\n".join(hits)
        system_message = {
            "role": "system",
            "content": f"Here is knowledge:\n{context}\n----------------------\nUse this knowledge ONLY IF IT IS RELEVANT to answer the user's query. If the user asks something irrelevant (outside the document), reply normally and do not mention the document. Give answer in one line and don't confuse the user by multiple answers. Talk in a friendly way."
        }
        messages = [system_message, {"role": "user", "content": user_input}]
    else:
        messages = [
            {"role": "system", "content": "Answer in a friendly, single-line way."},
            {"role": "user", "content": user_input}
        ]

    try:
        # Generate a response using Mistral AI
        chat_response = client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=300
        )

        # Extract and return the response
        response_content = chat_response.choices[0].message.content
        logging.info(f"Response: {response_content}")

        # Update message history (simplified for now)
        message_history.append({"role": "user", "content": user_input})
        message_history.append({"role": "assistant", "content": response_content})

        return jsonify({"response": response_content})
    except Exception as e:
        logging.error(f"Error in chat completion: {e}")
        return jsonify({"error": "Failed to process the request"}), 500
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)



