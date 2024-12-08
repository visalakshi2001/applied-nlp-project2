# Common Sense Reasoning

## Description
This project explores the common sense reasoning capabilities of language models using tasks from the BIG-Bench dataset. It evaluates models like DeBERTa-v3-base, GPT-3.5, GPT-4o-mini, and CommandR in zero-shot and fine-tuned settings, focusing on tasks such as Riddle-Sense, Causal-Judgement, and Odd-One-Out. Fine-tuning on CommonsenseQA is performed to investigate performance improvements across these reasoning tasks.

## Features
Zero-Shot Evaluation: Assesses multiple models on common sense reasoning tasks without task-specific training.
Fine-Tuning Integration: Implements fine-tuning on the CommonsenseQA dataset to improve model performance.
Task Diversity: Focuses on reasoning tasks like Riddle-Sense, Causal-Judgement, and Odd-One-Out from the BIG-Bench dataset.
Multi-Model Comparison: Compares the performance of DeBERTa-v3-base, GPT-3.5, GPT-4o-mini, and CommandR models.
Comprehensive Evaluation: Includes accuracy-based performance metrics and detailed task-specific analysis.

### Prerequisites
- Python 3.10.12
- pip 24.2
- virtualenv (highly recommended)
- packages listed in the `requirements.txt` file
- I recommend running the code given in the `notebooks/` folder in [Google Colab](https://colab.research.google.com/), as setting up this system and running it in local environment is seldom not possible. 

### Setup and Run
After cloning the repository into an environment, install the required packages using the following command on your terminal in the directory of the project:
```
pip install -r requirements.txt
```

Then open the ipynb notebooks in a colab environment, the notebooks have the following code:
1. `BigBench_Data_Exploration.ipynb`: Has the code to explore the BIG-Bench dataset and then evaluating models on zero-shot prompting
2. `CSQA_Deberta_Finetuning.ipynb`: Has the code for finetuning DeBERTa-v3 on CommonSenseQA dataset


### File and Folder Contents

1. `notebooks/`: contains all the jupyter notebooks used in the project
2. `responses/`: contains the responses of zero-shot prompting from all the models
3. `results/`: contains the evaluation results from the fine-tuning process
4. `finetune_commonsenseqa.py`: contains the same code from zero-shot prompting used in the notebook `BigBench_Data_Exploration.ipynb`
5. `zero_shot_bigbench.py`: contains the same code from  used in the notebook `CSQA_Deberta_Finetuning.ipynb`

## Write-up

See PDF file: `Project2_CommonSenseReasoning_BIGBench.pdf` for write-up