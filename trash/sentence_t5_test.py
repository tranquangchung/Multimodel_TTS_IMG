from sentence_transformers import SentenceTransformer, util
import torch

# Load model
model = SentenceTransformer('sentence-transformers/sentence-t5-base')

# Sentences
sentences = [
    "Keep the read centered on **WHISPER**.",
    "Maintain the focus on **WHISPER**.",
    "Consistently voice **ANGER**.",
    "Deliver this line in **GUILT**.",
    "Deliver this line in **WHISPER**."
]

# Encode
with torch.no_grad():
    embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine similarity
similarity_matrix = util.cos_sim(embeddings, embeddings)

print(similarity_matrix)
