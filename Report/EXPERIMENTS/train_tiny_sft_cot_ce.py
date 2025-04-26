import re
import torch
import ast
import operator as op
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
# from transformers import XLNetLMHeadModel  # Uncomment if you plan to support xlnet models
from sympy import Eq, solve
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


# CODE EVALUATOR
def evaluate_expression(expr):
    """
    evaluates an arithmetic expression given as a string or solves an equation if one is provided
    """


    if "=" in expr:
        # equation mode


        transformations = standard_transformations + (implicit_multiplication_application,)
        left_str, right_str = expr.split("=", 1)
        left_expr = parse_expr(left_str.strip(), transformations=transformations)
        right_expr = parse_expr(right_str.strip(), transformations=transformations)
        equation = Eq(left_expr, right_expr)

        free_symbols = list(equation.free_symbols)

        # if there is one free symbol, return its solution directly.
        if len(free_symbols) == 1:
            symbol = free_symbols[0]
            sol = solve(equation, symbol)
            return sol[0] if len(sol) == 1 else sol
        else:
            # for multiple variables, return the complete solution set.
            sol = solve(equation)
            return sol
    else:
        # arithmetic evaluation mode using AST
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.Mod: op.mod,
        }
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                return operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](_eval(node.operand))
            else:
                raise TypeError(f"Unsupported expression: {node}")

        parsed_expr = ast.parse(expr, mode='eval').body
        return _eval(parsed_expr)

# extraction function
def extract_final_expression(text):
    """
    extract expression
    """
    matches = re.findall(r'<<(.*?)>>', text, re.DOTALL)
    allowed_pattern = re.compile(r'^[\d+\-*/().= ]+$')
    valid_expressions = []
    for m in matches:
        candidate = m.strip()
        if '>' in candidate:
            candidate = candidate.split('>')[0].strip()
        if allowed_pattern.match(candidate):
            valid_expressions.append(candidate)
    if valid_expressions:
        return valid_expressions[-1]
    return ""

# data preprocessing
gsm8k = load_dataset("addaia/gsm8k-v2")#con fernando

def preprocess_gsm8k(dataset):
    processed = []
    for example in dataset:
        question = example["question"]
        answer_text = example["answer"]
        parts = answer_text.split("####")
        if len(parts) >= 2:
            reasoning = parts[0].strip()
            final_answer = parts[-1].strip()
            processed.append({
                "question": question,
                "reasoning": reasoning,
                "answer": final_answer,
                "code": example["code"]
            })
    return processed

train_data = preprocess_gsm8k(gsm8k["train"])
test_data  = preprocess_gsm8k(gsm8k["test"])

train_dataset = Dataset.from_list(train_data)
eval_dataset  = Dataset.from_list(test_data)

# 10% to debug faster
train_dataset_sampled = train_dataset
eval_dataset_sampled  = eval_dataset

# model and tokenizer setup
selected_model = "arnir0/Tiny-LLM"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(selected_model)
config = AutoConfig.from_pretrained(selected_model)

# load the model based on the model type
if config.is_encoder_decoder:
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(selected_model)
elif config.model_type == "xlnet":
    from transformers import XLNetLMHeadModel
    model = XLNetLMHeadModel.from_pretrained(selected_model)
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(selected_model)

model.eval()

# add a pad token if none exists to avoid warnings.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
model.to(device)

# train prompt
def format_train(example):
    target = example["code"] if example.get("code", "").strip() != "" else example["answer"]
    return (
        f"Question: {example['question']}\n"
        f"Chain-of-thought: {example['reasoning']}\n"
        f"Instruction: Provide ONLY the final arithmetic expression (using only numbers, operators, parentheses, and optionally an equals sign) that evaluates to the correct answer. Your answer MUST be enclosed in << and >>, with no additional commentary.\n"
        f"Final Expression: <<{target}>>"
    )

def tokenise_func(example):
    text = format_train(example)
    return tokenizer(text, truncation=True, max_length=256)

train_dataset_hf = train_dataset_sampled.map(tokenise_func, batched=False)
eval_dataset_hf  = eval_dataset_sampled.map(tokenise_func, batched=False)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# eval func
def compute_accuracy(model, dataset, tokenizer, debug=False):
    model.eval()
    correct = 0
    total = len(dataset)
    
    allowed_pattern = re.compile(r'^[\d+\-*/().= ]+$')
    
    for example in dataset:
        prompt = (
            f"Question: {example['question']}\n"
            f"Instruction: Provide ONLY the final arithmetic expression (using only numbers, operators, parentheses, and optionally an equals sign) that evaluates to the correct answer. Your answer MUST be enclosed in << and >>, with no additional commentary.\n"
            f"Final Expression:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            num_beams=5
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        expr = extract_final_expression(generated_text)
        
        if not allowed_pattern.match(expr) or expr == "":
            if debug:
                print("Filtered out invalid expression:", expr)
            continue
        
        try:
            predicted_value = evaluate_expression(expr)
            target_expr = example["code"] if example.get("code", "").strip() != "" else example["answer"]
            expected_value = evaluate_expression(target_expr)
            if abs(float(predicted_value) - float(expected_value)) < 1e-5:
                correct += 1
        except Exception as e:
            if debug:
                print("Exception evaluating:", e, "for expression:", expr)
    return correct / total if total > 0 else 0

# get accuracy per epoch
class EvalAccuracyCallback(TrainerCallback):
    def __init__(self, train_dataset, eval_dataset, tokenizer):
        self.train_dataset = train_dataset  # non-tokenised examples
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        init_train_acc = compute_accuracy(model, self.train_dataset, self.tokenizer, debug=False)
        init_eval_acc = compute_accuracy(model, self.eval_dataset, self.tokenizer, debug=False)
        print(f"Epoch 0: Train Accuracy = {init_train_acc:.4f}, Eval Accuracy = {init_eval_acc:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        train_acc = compute_accuracy(model, self.train_dataset, self.tokenizer, debug=False)
        eval_acc = compute_accuracy(model, self.eval_dataset, self.tokenizer, debug=False)
        epoch_num = int(state.epoch)
        print(f"Epoch {epoch_num}: Train Accuracy = {train_acc:.4f}, Eval Accuracy = {eval_acc:.4f}")

accuracy_callback = EvalAccuracyCallback(train_dataset_sampled, eval_dataset_sampled, tokenizer)

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir="./tiny_sft_cot_ce",
    num_train_epochs=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=5e-5,
    save_total_limit=1,
    report_to=[]
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=eval_dataset_hf,
    data_collator=data_collator,
    callbacks=[accuracy_callback]
)

# TRAIN
trainer.train()

# final eval
final_acc = compute_accuracy(model, eval_dataset_sampled, tokenizer)
print("Final accuracy:", final_acc)

model.save_model("tiny_sft_cot_ce")


