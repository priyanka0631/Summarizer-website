
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

def initialize_model():
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer, device



def efficient_summarization(text):
    model, tokenizer, device = initialize_model()
    input_text = "summarize:" + text
    tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
    summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
    summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
    return summary