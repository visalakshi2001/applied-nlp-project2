import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import seqio
import tensorflow_datasets as tfds
from bigbench.bbseqio import task_api
from bigbench.bbseqio import tasks
import re
import string
import cohere
import openai
import json
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMultipleChoice
import torch

co = cohere.ClientV2(api_key='COHERE_API_KEY')
client = openai.OpenAI(api_key='OPENAI_API_KEY')


def load_bbseqio_task(taskname):

    task = seqio.get_mixture_or_task("bigbench:causal_judgment.mul.t5_default_vocab.0_shot.all_examples")

    sequence_length = {"inputs": 32, "targets": 32}
    ds = task.get_dataset(
        sequence_length=sequence_length,  #  Trim long token sequences.
        split="all",  # Available splits = "all", "train", "validation".
    )
    for d in tfds.as_numpy(ds.take(5)):
        print(d)

    return ds

ds_riddle = load_bbseqio_task("bigbench:riddle_sense.mul.t5_default_vocab.0_shot.all_examples")
ds_ooo = load_bbseqio_task("bigbench:odd_one_out.mul.t5_default_vocab.0_shot.all_examples")
ds_cj = load_bbseqio_task("bigbench:causal_judgment.mul.t5_default_vocab.0_shot.all_examples")

def transform_question_string(question, choices):
    choices = [string.ascii_uppercase[i] + ": " + choice for i,choice in enumerate(choices)]
    inputs = question + "\n" + "\n".join(choices) + " \n" + "Answer: "

    return inputs

def extract_elements_from_full_question(example_dict, task):

    full_question = example_dict['inputs_pretokenized'].decode("utf-8")

    if task == 'riddle':
        question = re.findall("Q:\s(.*)\s+choice:(\s.*){3,}", full_question)[0][0]
        choices = re.findall("choice:\s(.*)", full_question)
    elif task == 'ooo':
        question = "Pick the odd word out:"
        choices = re.findall(f"{question}\s(.*)", full_question)[0].split(',')
    elif task == 'cj':
        question = re.findall("Q:\s(.*)\s.*", full_question)[0]
        choices = ["Yes", "No"]
    correct_answer = example_dict['answers'][0].decode('utf-8')

    return full_question, question, choices, correct_answer



# Load model and tokenizer
deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def get_deberta_response(question, choices):

    inputs = transform_question_string(question, choices)
    encodings = deberta_tokenizer(inputs, padding=True, return_tensors="pt")
    outputs = deberta_model(**encodings)
    scores = outputs.logits.softmax(dim=-1).detach().numpy()
    best_choice_idx = scores.argmax(axis=0)[1]
    return choices[best_choice_idx]


def get_cohere_response(question, choices, **kwargs):

    prompt = transform_question_string(question, choices)
    system_prompt = """
                    You are a helpful assistant. You have to solve the given common sense questions in the prompt.
                    After the questions there will be choices/options that you have to choose from.
                    Think logically before answering each question.
                    Output your response in this template 
                    choice_letter_of_correct_answer: answer_corresponding_to_the_letter
                    For example:
                        C: land
                    
                    Follow the template of response strictly. Choose your answer from the list choices given after the question.
                    Each choice will be preceeded by an uppercase letter
                    """
    model="command-r-08-2024"

    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt}]


    kwargs['model'] = model
    kwargs['messages'] = messages
    

    response = co.chat(**kwargs)

    return response.message.content[0].text


def get_gpt_response(question, choices, model_num='4-o-mini', **kwargs):
    
    prompt = transform_question_string(question, choices)
    system_prompt = """
                    You are a helpful assistant. You have to solve the given common sense questions in the prompt.
                    After the questions there will be choices/options that you have to choose from.
                    Think logically before answering each question.
                    Output your response in this template 
                    choice_letter_of_correct_answer: answer_corresponding_to_the_letter
                    For example:
                        C: land
                    
                    Follow the template of response strictly. 
                    """

    if model_num == "3.5":
        model="gpt-3.5-turbo"
    elif model_num == "4-o-mini":
        model="gpt-4o-mini"
    messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]

    kwargs["temperature"] = 0.0
    kwargs["top_p"] = 1
    kwargs["frequency_penalty"] = 0.0
    kwargs["presence_penalty"] = 0.0
    kwargs["timeout"] = 4*10*60  # 40 minutes
    kwargs["model"] = model
    kwargs["messages"] = messages


    response = client.chat.completions.create(**kwargs)

    return response.choices[0].message.content.strip()

def zero_shot_responses(ds, task, savefile, save=False):

    responses = []
    start = time.time()
    minute = time.time()

    for i, example in enumerate(tfds.as_numpy(ds)):
        
        full_question, question, choices, correct_answer = extract_elements_from_full_question(example, task)

        deberta_response = get_deberta_response(question, choices)
        gpt3_5_response = get_gpt_response(question, choices, model_num='3.5')
        gpt4_o_response = get_gpt_response(question, choices, model_num='4-o-mini')

        try:
            command_r_response = get_cohere_response(question, choices)
        except:
            print("Too many responses in a minute")
            time.sleep(60)
            try:
                command_r_response = get_cohere_response(question, choices)
            except:
                print(f"skipped index {i}")

        end = time.time()
        time_elapsed = end-start

        response = {
            "question": question,
            "choices": choices,
            "answer": correct_answer,
            "deberta": deberta_response,
            "gpt3.5": gpt3_5_response,
            "gpt4o": gpt4_o_response,
            "commandr": command_r_response
        }

        responses.append(response)


        if i%20 == 0:
            print(f"Generated [{i}/{len(ds)}] responses  |  Time elapses: {time_elapsed:.2f}")

        if save:
            with open(savefile, "a", encoding='utf-8') as f:
                f.write(json.dumps(response) + '\n')

    print(f"Responses saved to {savefile}")
        
    
    return responses

responses_riddle = zero_shot_responses(ds_cj, 'riddle', "responses/zero_shot_responses_riddle_sense.jsonl", save=True)
responses_cj = zero_shot_responses(ds_cj, 'cj', "response/zero_shot_responses_causal_judgement.jsonl", save=True)
responses_ooo = zero_shot_responses(ds_cj, 'ooo', "response/zero_shot_responses_oddoneout.jsonl", save=True)

