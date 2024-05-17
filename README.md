# tailor-cgo

This repo is for our NAACL Findings 2024 paper ["Tailoring Vaccine Messaging with Common-Ground Opinions"](URL-not-available-yet). We provide the data download scripts for both human-labeled and LLM-labeled responses, model weights for the BERT automatic evaluators, and inference scripts to assess how well-tailored a response of interest is.

We describe the dataset here (#Tailor-CGO Dataset), along with a tutorial of how to run automatic evaluation (#Automatic Evaluation).
This work builds on lots of previous work, mentioned in the last section (#Citations).

# `Tailor-CGO` Dataset
The dataset contains both human- and LLM-annotated preferences/scores for how "well tailored" each written response is. Annotations are structured as a (1) relative preference between two responses or (2) an absolute score given to each response individually.

To use the dataset:
```python
from datasets import load_dataset

# get either absolute or relative datasets:
dataset = load_dataset("DukeNLP/tailor-cgo", "relative_preferences")
dataset = load_dataset("DukeNLP/tailor-cgo", "absolute_scores")
```
You can view the full datacard on huggingface: https://huggingface.co/datasets/DukeNLP/tailor-cgo

Datapoints have the following structure:
```JSON
// Example of absolute score annotation
{
 "response_id": 96, 
 "concern": {
             "concern_id": 606, 
             "text": "the harmful ingredients in the influenza vaccine could..."
             },
 "opinion": {
             "opinion_id": 1108,
             "text": "When advocating for a bigger government..."
             },
 "system": {
            "model": "vicuna-33b-v1.3", 
            "temperature": 0.31619653,
            "prompt": "prompt-cot-health_expert-unguided"
            }, 
 "response": "I understand ...", 
 "evaluation": {
                "model": "gpt-4-1106-preview",  // 'crowdsourced' for human evaluated responses
                "temperature": 1.0, // None for human evaluated responses
                "prompt": "modified-geval",  // None for human evaluated responses
                "n_scores": 100,
                "raw_outputs": ["2\n\nThe response attempts to", 
                                "Tailoring Score = 1", ...],  // None for human evaluated responses
                "scores": [2, 1, ...], 
                "mean_score": 1.32, 
                "mode_score": 1,  // None for human evaluated responses
                }
}
```

```JSON
// Example of relative preference annotation
{
 "responseA": {
               "response_id": 0,
               "concern": {
                           "concern_id": 481,
                           "text": "we might be underestimating..."
                           },
               "opinion": {
                           "opinion_id": 56,
                           "text": "It is okay to..."
                           },
               "system": {
                          "model": "gpt-4-0613",
                          "temperature": 0.9046691,
                          "prompt": "prompt-cot-ai_assistant-unguided"
                          },
               "response": "I appreciate your..."
               },
 "responseB": {
               "response_id": 1,
               "concern": {
                           "concern_id": 481,
                           "text": "we might be underestimating..."
                           }, 
               "opinion": { // Note: opinion is not always the same as in A
                           "opinion_id": 56, "text": "It is okay to..."
                           },
               "system": { // Note: system is not always the same as in A
                          "model": "gpt-4-0613",
                          "temperature": 0.9046691,
                          "prompt": "prompt-cot-ai_assistant-unguided"
                          },
               "response": "I completely understand..."
               },
 "preferences": ["A", "A", "A"],
 "majority_vote": "A"
 }
```

On hugginface, the dataset has this structure. There are a few additional files that are not loaded through huggingface dataset loader. `*-relative_by_absolute` indicates absolute scores have been formatted as relative preference annotations by directly comparing scores. `*-relative_by_relative` indicates relative preferences are generated directly from prompting models through pairwise comparisons. 
```
data/
├── human_labeled/
│   ├── absolute_scores/
│   │   └── dev-absolute.jsonl
│   └── relative_preferences/
│       ├── dev-relative.jsonl
│       ├── dev-relative_by_absolute.jsonl
│       ├── dev-relative-extra.jsonl
│       ├── test-relative.jsonl
│       ├── train-relative.jsonl
│       └── train-relative-extra.jsonl
└── llm_labeled/
    ├── dev-relative_by_absolute.jsonl
    ├── dev-relative_by_relative.jsonl
    ├── test-relative_by_absolute.jsonl
    ├── test-relative_by_relative.jsonl
    ├── train-relative_by_absolute.jsonl
    └── train-absolute.jsonl
```

For each file, we list some relevant statistics on various definitions of their size $N$:
| file                            | unique responses | comparisons | annotations per sample |
|---------------------------------|------------------|-------------|------------------------|
| dev-absolute.jsonl              | 400              | N/A         | 3                      |
| dev-relative_by_absolute.jsonl* | 400              | 200         | 3                      |
| dev-relative-extra.jsonl**      | 600              | 400         | 1                      |
| dev-relative.jsonl              | 400              | 200         | 3                      |
| test-relative-extra.jsonl**     | 1200             | 800         | 1                      |
| test-relative.jsonl             | 800              | 400         | 3                      |
| train-absolute.jsonl            | 20000            | N/A         | 100                    |
| train-relative.jsonl            | 1200             | 600         | 1                      |

NOTE:
> *This file is translated from absolute scores to relative comparisons by comparing scores across responses in `dev-absolute.jsonl`.
> 
> **Unlike rest of files, these comparisons are non-iid. We do not use these files in our paper but make them available here.

For further explanation of how the data is collected, please see [our paper](URL-not-available-yet).

# Automatic Evaluation
We release the model weights for our automatic evaluators, both $BERT_{REL}$ and $BERT_{ABS}$. Below is a demonstration of how to use these models:

Download and open our latest release. This will download model weights and place them in the `local/` directory. Then, generate predictions according to this demo below:

```python
from models.regression import Finetuner
from transformers import BertTokenizer

# load model from our github release
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = Finetuner.load_from_checkpoint('local/regression-fine-2e-ckpt/last-release.ckpt')
model.eval()

# data to process
concern = "there seems to be a lack of proper blinding and control treatments in this vaccine research which could potentially skew the results"
opinion = "The idea that robots and computers will be able to do most of the jobs currently done by humans in the future seems extremely realistic."
message = "Certainly, it's understandable to be concerned about..."
# format for passing to model
data = 'SEP'.join((message, concern, opinion))

# predict
response = tokenizer(data, 
                     padding=True, 
                     truncation=True, 
                     max_length=500, 
                     return_tensors='pt')
scores = model.forward({
        'response': response,
        'labels': None,
    }, predict=True)['scores'].tolist()

# result
print(scores)
```

# Citations
If you use this content in your work, please cite our paper:

```bibtex
// TBD: not on arxiv yet
```

If you mention the taxonomy of vaccine concerns (VaxConcerns) which the dataset is based on, please cite this paper:

```bibtex
@article{VaxConcerns,
	title = {Development and validation of {VaxConcerns}: A taxonomy of vaccine concerns and misinformation with Crowdsource-Viability},
	issn = {0264-410X},
	url = {https://www.sciencedirect.com/science/article/pii/S0264410X2400255X},
	doi = {https://doi.org/10.1016/j.vaccine.2024.02.081},
	journal = {Vaccine},
	author = {Stureborg, Rickard and Nichols, Jenna and Dhingra, Bhuwan and Yang, Jun and Orenstein, Walter and Bednarczyk, Robert A. and Vasudevan, Lavanya},
	year = {2024},
}
```
