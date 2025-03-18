import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1"
tokenizer = AutoTokenizer.from_pretrained(Qwen2.5-Max)
model = AutoModelForCausalLM.from_pretrained(Qwen2.5-Max)

# Quantize the model (example using FP16)
model.half()  # Convert model to FP16

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize the input text efficiently
def tokenize_text(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs

# Optimize inference
def generate_response(prompt):
    inputs = tokenize_text([prompt])
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=16384,
            temperature=0.785,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
