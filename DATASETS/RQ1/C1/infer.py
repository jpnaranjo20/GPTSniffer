import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = './saved_model'

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()


def predict(code: str) -> str:
    inputs = tokenizer(code, padding='max_length', max_length=512,
                       truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    label = torch.argmax(logits, dim=1).item()
    return 'ChatGPT' if label == 1 else 'Human'


if len(sys.argv) < 2:
    print("Usage: python infer.py path/to/file.java [file2.java ...]")
    sys.exit(1)

for path in sys.argv[1:]:
    with open(path, 'r') as f:
        code = f.read()
    result = predict(code)
    print(f"{path}: {result}")
