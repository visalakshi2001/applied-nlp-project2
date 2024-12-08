import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("commonsense_qa")

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name, num_labels=5)
model.to(device)

def preprocess_function(examples):
    questions = examples['question']
    choices = examples['choices']
    num_choices = len(choices[0]['text'])

    flat_choices = [choice for choice_set in choices for choice in choice_set['text']]

    repeated_questions = [[q] * num_choices for q in questions]
    flat_questions = [q for sublist in repeated_questions for q in sublist]

    inputs = [f"Question: {q} Choice: {c}" for q, c in zip(flat_questions, flat_choices)]

    tokenized_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=128)

    tokenized_inputs = {k: [v[i:i+num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_inputs.items()}

    labels = []
    for ans in examples['answerKey']:
        if ans and 'A' <= ans <= 'E':  
            labels.append(ord(ans) - ord('A'))
        else:
            labels.append(0)  

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

encoded_dataset = dataset.map(preprocess_function, batched=True)
def finetune_model(model, train_flag=True):

    if train_flag:
        training_args = TrainingArguments(
            output_dir="./deberta-finetuned",  
            evaluation_strategy="epoch",     
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            save_total_limit=2,             
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['validation'],
            tokenizer=tokenizer,
        )

        trainer.train()
        
        results = trainer.evaluate(encoded_dataset['validation'])
        print(f"Validation Results: {results}")
    
    return model


def predict_answer(model, tokenizer, question, choices):
    inputs = [f"Question: {question} Choice: {choice}" for choice in choices]

    encoding = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # The model expects inputs in shape: (batch_size, num_choices, seq_length)
    # Here we have one "batch" with multiple choices, so we add a batch dimension
    for key in encoding:
        encoding[key] = encoding[key].unsqueeze(0)  # Now shape: (1, num_choices, seq_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)


    predicted_label = outputs.logits.argmax(dim=1).item()
    predicted_answer = choices[predicted_label]

    return predicted_answer

def calculate_validation_score(model, tokenizer, dataset):
    total_correct = 0
    total_questions = len(dataset['validation'])
    incorrect_instances = []

    for instance in tqdm(dataset['validation'], desc="Validating"):
        question = instance['question']
        choices = instance['choices']['text']
        correct_answer = instance['answerKey']
        correct_answer = choices[ord(correct_answer) - ord('A')]
        predicted_answer = predict_answer(model, tokenizer, question, choices)

        if predicted_answer == correct_answer or predicted_answer in correct_answer:
            total_correct += 1
        else:
            incorrect_instances.append(instance)

    validation_score = total_correct / total_questions

    return validation_score, incorrect_instances

val_score, incorrect_ins = calculate_validation_score(model, tokenizer, dataset)
print(f"Total Correct Answers: {val_score*len(dataset['validation'])}")
print(f"Total Questions: {len(dataset['validation'])}")
print(f"Incorrect Answers: {round((1-val_score)*len(dataset['validation']))}")
print(f"Validation Score: {val_score*100:.2f}")

model.save_pretrained("./final_model/deberta-finetuned")
tokenizer.save_pretrained("./final_model/deberta-finetuned")