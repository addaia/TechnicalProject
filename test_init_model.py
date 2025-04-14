!/usr/bin/env python
import time
import torch
import re
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    XLNetLMHeadModel
)


gsm8k = load_dataset("addaia/gsm8k-v2")

def preprocess(example):
    parts = example["answer"].split("####")
    if len(parts) >= 2:
        reasoning = parts[0].strip()
        answer = parts[-1].strip()
    else:
        reasoning = ""
        answer = example["answer"].strip()
    return {
        "question": example["question"],
        "reasoning": reasoning,
        "answer": answer
    }

# load
eval_data = [preprocess(x) for x in gsm8k["test"]]
eval_dataset = Dataset.from_list(eval_data)

# model
selected_model = "arnir0/Tiny-LLM" # or "distilgpt2" for other model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(selected_model)
config = AutoConfig.from_pretrained(selected_model)

# select approrpiate class
if config.is_encoder_decoder:
    model = AutoModelForSeq2SeqLM.from_pretrained(selected_model)
elif config.model_type == "xlnet":
    model = XLNetLMHeadModel.from_pretrained(selected_model)
else:
    model = AutoModelForCausalLM.from_pretrained(selected_model)

# pad
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()

# prompt format
def format_eval(example):
    return (
        f"Question: {example['question']}\n"
        "Answer:"
    )

def generate_answer(example):
    prompt = format_eval(example)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # generate
    output = model.generate(
        **inputs,
        max_new_tokens=50, 
        num_return_sequences=1,
        do_sample=False  
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # extract
    if "Answer:" in generated_text:
        pred_answer = generated_text.split("Answer:")[-1].strip()
    else:
        pred_answer = generated_text.strip()
    

    numbers = re.findall(r'\d+', pred_answer)
    if numbers:
        pred_answer = numbers[-1]
    else:
        pred_answer = ""  
    return pred_answer

# evaluation
correct = 0
total = len(eval_dataset)
total_inference_time = 0.0


for i, example in enumerate(eval_dataset):
    print(i)

    
    true_answer = example["answer"].strip()

    start_time = time.perf_counter()
    pred_answer = generate_answer(example)
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    total_inference_time += inference_time

    
    # extract
    true_numbers = re.findall(r'\d+', true_answer)
    true_num = true_numbers[-1] if true_numbers else ""
    
    if pred_answer == true_num:
        correct += 1

accuracy = (correct / total) * 100
average_inference_time = total_inference_time / total

print("accuracy: {:.2f}%".format(accuracy))
print("time ineference: {:.4f} seconds".format(average_inference_time))
