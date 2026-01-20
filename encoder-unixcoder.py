from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained("")
model = AutoModelForSequenceClassification.from_pretrained("")


code_snippet = """
void process(char *input) {
    char buffer[50];
    strcpy(buffer, input); // Potential buffer overflow
}
"""

# Tokenize the input
inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions, dim=1).item()

# Output the result
print("Vulnerable Code" if predicted_label == 1 else "Safe Code")
