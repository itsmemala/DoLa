import os
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import json
import jsonlines
import random
# import llama
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import argparse
# from transformers import BitsAndBytesConfig, GenerationConfig
# from peft import PeftModel
# from peft.tuners.lora import LoraLayer
import evaluate

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'hl_llama_7B': 'huggyllama/llama-7b',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'flan_33B': 'timdettmers/qlora-flan-33b'
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='nq_open', help='dataset for querying the model')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument("--file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    print('Loading model..')
    # tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    # model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    model_name = MODEL
    kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
    kwargs["device_map"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
    model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True, **kwargs).to(device)
    device = "cuda"

    print('Loading model responses..')
    prompts, responses = [], []
    if 'baseline' in args.file_name:
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.file_name}.json', 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            prompts.append(data['full_input_text'][i])
            response = data['model_completion'][i] if 'strqa' in args.dataset_name else data['model_answer'][i] # For strqa, we want full COT response
            responses.append(response)
    else:
        # data = []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.file_name}.json', 'r') as read_file:
            for line in read_file:
                # data.append(json.loads(line))
                data = json.loads(line)
                prompts.append(data['prompt'])
                responses.append(data['response1'])

    
    print('Getting token probability scores..')
    # Get token probabilities
    scores = []
    # for i,sample in tqdm(enumerate(data)):
        # prompt = sample['prompt']
        # response = sample['response1']
    for prompt,response in tqdm(zip(prompts,responses)):
        tokenized_input = tokenizer([prompt+response], return_tensors = 'pt').input_ids.to(device)
        # tokenized_input = tokenized_input[tokenized_input != tokenizer.pad_token_id]
        tokenized_prompt = tokenizer([prompt], return_tensors = 'pt').input_ids
        # tokenized_prompt = tokenized_prompt[tokenized_prompt != tokenizer.pad_token_id]
        # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
        target_ids = tokenized_input.clone().to(device)
        target_ids[0][:len(tokenized_prompt[0])] = -100
        # model_output = model(torch.reshape(tokenized_input, (1, -1)), labels=target_ids, output_hidden_states=True)
        model_output = model(tokenized_input, labels=target_ids, output_hidden_states=True)
        average_neg_log_likelihood = model_output['loss'] # len normalised predictive entropy
        neg_log_likelihood = average_neg_log_likelihood * (len(tokenized_input[0]) - len(tokenized_prompt[0])) # sequence predictive entropy
        scores.append(np.array([average_neg_log_likelihood.detach().cpu().numpy(),neg_log_likelihood.detach().cpu().numpy()]))

    print('Saving token probability scores..')
    np.save(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.file_name}_uncertainty_scores.npy', scores)
    

if __name__ == '__main__':
    main()