# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py

import re
import os
import json
import random
import torch
import numpy as np
from datasets import load_dataset
import evaluate
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request

from dola import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"

rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")
        

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def build_prompt(input_text, n_shot, cot_flag, shuffle):
    # demo = create_demo_text(n_shot, cot_flag, shuffle)

    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b") # "baffo32/decapoda-research-llama-7B-hf"
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./gsm8k")
    parser.add_argument("--output-path", type=str, default="./gsm8k_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--early-exit-w-probe", type=bool, default=False)
    parser.add_argument("--best_layers_file_path", type=str, default="")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    print('Loading data..')
    # Load data
    use_split = 'validation' # 'validation' / 'train'
    len_dataset = 1800 # 1800 / 5000
    start_at = 0
    hf_dataset_name = 'mandarjoshi/trivia_qa'
    dataset = load_dataset(hf_dataset_name, 'rc.nocontext', streaming= True)[use_split]
    prompts, all_ref_answers = [], []
    for idx,val in enumerate(list(dataset.take(len_dataset))[start_at:]):
        question = val['question']
        cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
        prompts.append(cur_prompt)
        all_ref_answers.append(val['answer'])
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:", "\end{code}"])
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1 and args.early_exit_w_probe == False:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
    elif len(early_exit_layers) == 2 and args.early_exit_w_probe == False:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "dola-static"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    elif args.early_exit_w_probe == False:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    else:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and probe based premature layers")
        best_layers = np.load(f'{args.best_layers_file_path}.npy')
    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    for idx,sample in enumerate(tqdm(prompts)):
        if args.early_exit_w_probe == True:
            # # candidate_premature_layers = best_layers[idx] + 1 # shift indexing from 0-31 to 1-32
            # # candidate_premature_layers = [layer for layer in candidate_premature_layers if layer!=32] # Exclude last layer
            # lower_most_layer = np.min(best_layers[idx] + 1) # shift indexing from 0-31 to 1-32
            # candidate_premature_layers = [layer for layer in range(lower_most_layer+1) if layer%2==0 and layer!=32]
            # premature_layer_dist = {l:0 for l in candidate_premature_layers}
            if best_layers[idx]!=-1:
                mode = "dola"
                mature_layer = early_exit_layers[-1]
                premature_layer = None
                if args.repetition_penalty is None:
                    args.repetition_penalty = 1.2
                # candidate_premature_layers = [best_layers[idx]]
                candidate_premature_layers = [j for j in range(best_layers[idx]+1)]
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
            if best_layers[idx]==-1: # No contrast; default decoding
                mode = "baseline"
                mature_layer = None
                premature_layer = None
                candidate_premature_layers = None
                if args.repetition_penalty is None:
                    args.repetition_penalty = 1.0
        # input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG, args.do_shuffle)
        input_text = sample
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
        # model_answer = clean_answer(model_completion)
        model_answer = model_completion
        checkgens = ['Q:','QA1:','QA2:','Q.', 'B:']
        for check_gen in checkgens: # Fix generation stopping errors
            model_answer = model_answer.split(check_gen)[0]   
        # print(input_text)
        # print(model_completion)
        print('\nModel answer:',model_answer)
        # if idx==10: break
        # is_cor = is_correct(model_answer, sample['output'])
        labels_dict = {'exact_match': 0.0,
                        'rouge1_to_target':0.0,
                        'rouge2_to_target':0.0,
                        'rougeL_to_target':0.0}
        reference_answers_unformatted = all_ref_answers[idx]
        reference_answers = reference_answers_unformatted['aliases'] + reference_answers_unformatted['normalized_aliases']
        for answer in reference_answers:
            predictions = [model_answer.lstrip()]
            references = [answer]
            results = exact_match_metric.compute(predictions=predictions,
                                                    references=references,
                                                    ignore_case=True,
                                                    ignore_punctuation=True)
            labels_dict['exact_match'] = max(results['exact_match'], labels_dict['exact_match'])
            rouge_results = rouge.compute(predictions=predictions, references=references)
            for rouge_type in ['rouge1','rouge2','rougeL']:
                labels_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type],
                                                                labels_dict[rouge_type + '_to_target'])
        is_cor = 1 if labels_dict['rouge1_to_target']>0.3 else 0
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
            # print(f'Question: {sample["instruction"]}\n\n'
            #     f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
            #     f'Model Answers: {model_answer}\n\n'
            #     f'Model Completion: {model_completion}\n\n'
            #     f'Is correct: {is_cor}\n\n')

        print(f'Num of total question: {len(answers)}, '
                f'correct num: {sum(answers)}, '
                f'correct rate: {float(sum(answers))/len(answers)}.')

    if mode == "dola"and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    print(f"{float(sum(answers))/len(answers)}")