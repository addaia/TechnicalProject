#!/usr/bin/env python
import re
import torch
import ast, operator as op
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# data loading and preprocessing
gsm8k = load_dataset("addaia/gsm8k-v2")

def preprocess(example):
    # split
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

train_data = [preprocess(x) for x in gsm8k["train"]]
eval_data  = [preprocess(x) for x in gsm8k["test"]]

train_dataset = Dataset.from_list(train_data)
eval_dataset  = Dataset.from_list(eval_data)

# model and tokenizer setup
selected_model = "distilgpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(selected_model)
config    = AutoConfig.from_pretrained(selected_model)
model     = AutoModelForCausalLM.from_pretrained(selected_model)
model.eval()

# add a pad token if none exists to avoid warnings.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
model.to(device)

# prompt formatting
def format_train(example):
    return (
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}"
    )

def tokenise_function(example):
    text = format_train(example)
    return tokenizer(text, truncation=True, max_length=256)

train_dataset_tokenised = train_dataset.map(tokenise_function, batched=False)
eval_dataset_tokenised  = eval_dataset.map(tokenise_function, batched=False)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# eval func
def compute_accuracy(model, dataset, tokenizer):
    model.eval()
    correct = 0
    total = len(dataset)
    for example in dataset:
        prompt = f"Question: {example['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_text = generated_text.split("Answer:")[-1].strip()
        numbers = re.findall(r"(-?\d+(?:\.\d+)?)", answer_text)
        predicted = numbers[-1] if numbers else None
        try:
            if predicted is not None and abs(float(predicted) - float(example["answer"])) < 1e-5:
                correct += 1
        except ValueError:
            pass
    return correct / total if total > 0 else 0

# class to track accuracy
class EvalAccuracyCallback(TrainerCallback):
    def __init__(self, train_dataset, eval_dataset, tokenizer):
        self.train_dataset = train_dataset  # non-tokenised examples
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.epochs = [0]
        self.train_accuracies = []
        self.eval_accuracies = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        init_train_acc = compute_accuracy(model, self.train_dataset, self.tokenizer)
        init_eval_acc = compute_accuracy(model, self.eval_dataset, self.tokenizer)
        self.train_accuracies.append(init_train_acc)
        self.eval_accuracies.append(init_eval_acc)
        print(f"Epoch 0: Train Accuracy = {init_train_acc:.4f}, Eval Accuracy = {init_eval_acc:.4f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        train_acc = compute_accuracy(model, self.train_dataset, self.tokenizer)
        eval_acc = compute_accuracy(model, self.eval_dataset, self.tokenizer)
        epoch_num = int(state.epoch)
        self.epochs.append(epoch_num)
        self.train_accuracies.append(train_acc)
        self.eval_accuracies.append(eval_acc)
        print(f"Epoch {epoch_num}: Train Accuracy = {train_acc:.4f}, Eval Accuracy = {eval_acc:.4f}")

accuracy_callback = EvalAccuracyCallback(train_dataset, eval_dataset, tokenizer)

# training setup
training_args = TrainingArguments(
    output_dir="./distilgpt_sft",
    num_train_epochs=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenised,
    eval_dataset=eval_dataset_tokenised,
    data_collator=data_collator,
    callbacks=[accuracy_callback]
)

# train and save the model (only weights)
trainer.train()
model.save_pretrained("final_model_distilgpt2_sft")


