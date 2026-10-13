from typing import Union

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from PIL import Image
from torchvision import transforms

from helper_lib.model import get_model
from helper_lib.utils import load_model
import io
import torch

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
]

cnn_model = load_model(get_model("AssignmentCNN"), "models/cnn_cifar10.pth")
cnn_model.eval()
preprocess = transforms.Compose([
    transforms.Resize((64,64)), transforms.ToTensor()
])


app = FastAPI()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {"class": CLASSES[idx], "confidence": float(probs[idx])}


# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.calculate_embedding(request.word)
    return {
        "word": request.word,
        "dimensions": len(embedding),
        "in_vocabulary": embedding_model.has_vector(request.word),
        "embedding": embedding,
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    similarity = embedding_model.calculate_similarity(request.word1, request.word2)
    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": similarity,
        "in_vocabulary": (
            embedding_model.has_vector(request.word1)
            and embedding_model.has_vector(request.word2)
        ),
    }
