import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# import huggingface transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

"""## Testing - CONCATENATION(ee+er)"""

def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support! 
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits

import pytorch_pretrained_bert
tokenizer = torch.load("/home/kshitij_1921cs23/Anubhab/Persona_Persuasion/special3_gpt2_tokenizer.pkl")

"""### Load Model"""

# load the model
device = 'cpu' 
#torch.device("cuda")
model_A = GPT2LMHeadModel.from_pretrained("gpt2")
model_B = GPT2LMHeadModel.from_pretrained("gpt2")


model_A_states, model_B_states = torch.load("PATH_TO_RPTCS_model")

model_A.load_state_dict(model_A_states)
#model_B.load_state_dict(model_B_states)

model_A.to(device)
#model_B.to(device)

model_A.eval()
#model_B.eval()


prev_input = tokenizer.encode("A:")
prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
# past_position_ids = torch.LongTensor([[0, 1]]).to(device)

temperature = 0.8
top_k = 400
top_p = 0.9



past = None
flag = True

sep = tokenizer.encode("\n\n\n")

while flag:
    "Sampling based method"
    sent = []
    with torch.no_grad():
        for i in range(200):
            logits, past = model_A(prev_input, past_key_values=past, return_dict=False)
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_k=200, top_p=0.9)
            # prev_input = logits.argmax(-1).unsqueeze(1)
            probs = F.softmax(logits, -1)
            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == 628:
                break
            elif prev_word == tokenizer.encode("[EOS]"):
                flag = False
                break
            else:
                sent.append(prev_word)
            

    if not flag:
        break

    print("A:" + tokenizer.decode(sent).split("<|endoftext|>")[0])
    
    # finish tail
    prev_input = torch.LongTensor(sep).unsqueeze(0).to(device)
    _, past = model_A(prev_input, past_key_values=past, return_dict=False)
    
    # input and update B's utterance
    user = input("B:")
    
    if user == "quit":
        break
        
    user = tokenizer.encode("B:" + user)
    prev_input = user + sep
    prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
    
    outputs = model_A(prev_input, past_key_values=past, return_dict=True)
    logits, past = outputs[0], outputs[1]

    suffix = tokenizer.encode("A:")
    prev_input = torch.LongTensor(suffix).unsqueeze(0).to(device)