# This will contain the code to test the model perfomance against few standard benchmark models
# Locally testing the model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#model's path
MODEL_PATH = "./models/my_babyLM_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test text
text = "The cat sat on"
inputs = tokenizer(text, return_tensors="pt").to(device)

# Generate next tokens
outputs = model.generate(**inputs, max_length=20)
print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))