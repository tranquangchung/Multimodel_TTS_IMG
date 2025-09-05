import torch
from transformers import CLIPProcessor, CLIPModel

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)
for param in clip_model.parameters():
    param.requires_grad = False
clip_model.eval().to(device)

# Sentences để test
sentences = [
    "Keep the read centered on **WHISPER**.",
    "Maintain the focus on **WHISPER**.",
    "Consistently voice **ANGER**.",
    "Deliver this line in **GUILT**.",
    "Deliver this line in **WHISPER**."
]

# Encode text với CLIP
inputs = clip_processor(text=sentences, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    text_features = clip_model.get_text_features(**inputs)  # [N, hidden_dim]

# Normalize embeddings
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Cosine similarity matrix
similarity_matrix = text_features @ text_features.T

print(similarity_matrix)
