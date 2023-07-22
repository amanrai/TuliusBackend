import fastapi
import uvicorn
from fastapi import FastAPI
from typing import Optional, List, Dict
from pydantic import BaseModel
from generator import Generator
from argparse import ArgumentParser
import torch
import json
from sentence_transformers import SentenceTransformer
import time

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="togethercomputer/RedPajama-INCITE-Chat-3B-v1")
parser.add_argument("--tokenizer", type=str, default="togethercomputer/RedPajama-INCITE-Chat-3B-v1")
parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--half_mode", type=str, default="fp16", choices=["fp16", "bf16"])
args = parser.parse_args()

print("Loading the Embedder...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(args.embedder).to("cuda")
sentence = ['This framework generates embeddings for each input sentence']
embedding = embedder.encode(sentence)
print(len(embedding[0]))

print("Starting Up...")

if (args.half_mode == "bf16"):
    half_mode = torch.bfloat16
else:
    half_mode = torch.float16

generator = Generator(args.model, args.model, half_mode=half_mode)

print("Model Loaded.")

class GeneratorInput(BaseModel):
    prompt: str
    generator_options: Dict

class EmbedderInput(BaseModel):
    chunk: str
    
app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Welcome to the Tulius Generator.",
        "lm":args.model,
        "embedder":args.embedder
    }

@app.post("/generate")
def generate(input: GeneratorInput):
    print(input.prompt)
    start = time.time()
    _out =  generator.generate(input.prompt, input.generator_options)
    print("Generation took:", time.time() - start)
    return _out

@app.post("/embed")
def embed(input: EmbedderInput):
    embedding =  embedder.encode([input.chunk])
    return json.dumps(embedding.tolist())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
