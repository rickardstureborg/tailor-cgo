from typing import List
import time
import json
import yaml
import re
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

random.seed(42)


def load_criteria():
    # load all .yaml files in the criteria folder
    criteria = dict()
    for file in Path("src/criteria").glob("*.yaml"):
        l = list(yaml.safe_load(file.read_text()).values())[0]
        for i, p in enumerate(l):
            k = file.stem[0] + '#' + str(i)
            v = p['description']
            assert k not in criteria
            criteria[k] = v

    return criteria

def sample_criteria(criteria, num_criteria=5):
    # return a list of criteria ids and a list of criteria descriptions
    criteria_ids = random.sample(list(criteria.keys()), num_criteria)
    criteria_descriptions = [criteria[cid] for cid in criteria_ids]
    return criteria_ids, criteria_descriptions


"""
role extraction
"""
def extract_info(s):
    match = re.search(r'cot-(.*?)-(un)?guided', s)
    return match.group(1) if match else None

"""
mapping
"""
def to_map(data_whole, key_col_name:str, val_col_name:str):
    """
    Convert the corresponding index and content column in a dataframe to a map.

    :param df: dataframe
    :param key_col_name: column name of the key
    :param val_col_name: column name of the value
    """
    df_whole = pd.DataFrame(data_whole)
    data = {
    'index_col': df_whole[key_col_name],
    'context_col':df_whole[val_col_name]
}
    df = pd.DataFrame(data)
    mapping = df.set_index('index_col')['context_col'].to_dict()

    return mapping


"""
all attributes
"""

def get_role(role_idx: str, my_map: dict) -> str:
    """
    Get an attribute for the given role.

    :param role_idx: index of the role
    """
    if role_idx in my_map:
        role_prompt = my_map[role_idx]
    else:
        print(f"Role {role_idx} not found in map")
    return role_prompt

def get_concern(concern_idx: int, concern_map: dict) -> str:
    """
    Get a concern for the given concern index.

    :param concern_idx: index of the concern
    """
    if concern_idx in concern_map:
        concern_prompt = concern_map[concern_idx]
    else:
        print("Concern not found in map")
    return concern_prompt.strip('.?')

def get_opinion(opinion_idx: int, opinion_map: dict) -> str:
    """
    Get an opinion for the given opinion index.

    :param opinion_idx: index of the opinion
    """
    if opinion_idx in opinion_map:
        opinion_prompt = opinion_map[opinion_idx]
    else:
        print("Opinion not found in map")
    # change the first letter to lowercase
    opinion_prompt = opinion_prompt[0].lower() + opinion_prompt[1:]
    return opinion_prompt.strip('.')


"""
get prompt
"""

def get_prompt( role_idx: str,
                role_map: dict,
                cot: bool,
                concern_idx: int,
                concern_map: dict,
                opinion_idx: int,
                opinion_map: dict,
                model: str,
                temperature: float,
                few_shot: List[int] = None,
                criteria: List[str] = None,) -> str:
    """
    Generate a prompt for the given role, with the given parameters.

    :param role_idx: str of the role
    :param cot: whether to use chain of thought
    """

    system_prompt = get_role(role_idx, role_map)
    if criteria:
        system_prompt += '\nIn the response, you follow these rules:\n' + '\n'.join(['- ' + c for c in criteria])
    concern = get_concern(concern_idx, concern_map)
    opinion = get_opinion(opinion_idx, opinion_map)

    GEN_RES_PROMPT = "Please write one short paragraph on vaccine intervention tailored to an individual's common ground opinion."

    GEN_COT_PROMPT = "Before giving the actual response, please first think step by step and concisely list the strategies you would use to convince the person to get vaccinated."

    GEN_AFTER_COT_PROMPT = "Considering what you have reasoned, please write one short paragraph to convine them to take vaccine."

    user_profile_prompt = ''.join([
        f"The person believes that {opinion}.",
        f" The person has a vaccine concern regarding {concern}.",
    ])

    if not cot:
        return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_profile_prompt + ' ' + GEN_RES_PROMPT},
            ]
    else:
        return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_profile_prompt + ' ' + GEN_COT_PROMPT},
                {"role": "user", "content": GEN_AFTER_COT_PROMPT},
            ]



if __name__ == "__main__":
    #load the data, mapping the index and the content for each attribute
    base_data_path = "./"
    recipes = pd.read_csv(base_data_path + "recipes_round2.csv")
    concern_map = to_map(pd.read_csv(base_data_path + "concerns.csv"), 'concern_id', 'statement')
    opinion_map = to_map(pd.read_csv(base_data_path + "opinions.csv"), 'opinion_id', 'statement')
    role_map = to_map(pd.read_csv(base_data_path + "roles.csv"), 'Role', 'Prompt')
    criteria_map = load_criteria()
    output = []
    output_criteria = []

    for i in tqdm(range(len(recipes))):
        curr_cot = '-cot-' in str(recipes['prompt_name'][i])
        curr_criteria = '-guided' in str(recipes['prompt_name'][i])
        curr_role = extract_info(str(recipes['prompt_name'][i]))
        curr_concern_idx = recipes['concern_id'][i]
        curr_opinion_idx = recipes['opinion_id'][i]
        curr_model = recipes['model'][i]
        curr_temperature = recipes['temperature'][i]
        if curr_criteria:
            curr_criterion_ids, curr_criterion_descriptions = sample_criteria(criteria_map)
        else:
            curr_criterion_ids = None
            curr_criterion_descriptions = None
        curr_output = get_prompt(curr_role, role_map, curr_cot, curr_concern_idx, concern_map, curr_opinion_idx, opinion_map, curr_model, curr_temperature, few_shot=None, criteria=curr_criterion_descriptions)
        output.append(curr_output)
        output_criteria.append(curr_criterion_ids)

    recipes['prompt'] = output
    recipes['criteria'] = output_criteria
    recipes.to_csv(base_data_path + "recipes_with_prompts_round2.csv", index=False)