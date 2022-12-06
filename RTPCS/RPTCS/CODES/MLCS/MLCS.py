"""### Import packages"""
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='GPU_NUMBER'


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

"""### Load Data"""

def extract_data(df_dialogs):
    data = []
    persona = []
    path = []
    topic =[]
    for i in tqdm.trange(len(df_dialogs)):
        if df_dialogs['speaker'][i] == 0:
            text = "A:" + str(df_dialogs["utterance"][i])
            persona.append(str(df_dialogs["persona"][i]))
            path.append(str(df_dialogs["path"][i]))
            topic.append(str(df_dialogs["topic"][i]))
            data.append(text)
        else:
            text = "B:" + str(df_dialogs["utterance"][i])
            persona.append(str(df_dialogs["persona"][i]))
            path.append(str(df_dialogs["path"][i]))
            topic.append(str(df_dialogs["topic"][i]))
            data.append(text)
    return data, persona, path, topic

df = pd.read_csv("DATASET_PATH")
print(df.describe())

data, persona, path, topic = extract_data(df)  

values=df['conv_id'].unique().tolist()
conv_ids = df['conv_id'].tolist()

# print a sample of the persona based topic shift dialog
dataset = []
conversation = []
personaset = []
persona_conversation = []
pathset = []
path_conversation = []
topicset = []
topic_conversation = []
for conv in values:
    for i in range(0, df.shape[0]):
        if(conv_ids[i]==conv):
            conversation.append(data[i])
            persona_conversation.append(persona[i])
            path_conversation.append(path[i])
            topic_conversation.append(topic[i])
        else:
            continue
    dataset.append(conversation)
    personaset.append(persona_conversation)
    pathset.append(path_conversation)
    topicset.append(topic_conversation)
    conversation = []
    persona_conversation = []
    path_conversation = []
    topic_conversation = []

data = dataset
persona = personaset
path = pathset
topic = topicset



indices = np.arange(len(data))
np.random.shuffle(indices)
train_data = [data[idx] for idx in indices[:'XXXX']] # XXXX: nuber of dialogues in train dataset
val_data = [data[idx] for idx in indices['XXXX':]]
train_persona = [persona[idx] for idx in indices[:'XXXX']]
val_persona = [persona[idx] for idx in indices['XXXX':]]
train_path = [path[idx] for idx in indices[:'XXXX']]
val_path = [path[idx] for idx in indices['XXXX':]]
train_topic = [topic[idx] for idx in indices[:'XXXX']]
val_topic = [topic[idx] for idx in indices['XXXX':]]


from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup


# load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class TopicShiftDataset(Dataset):
    def __init__(self, data, persona, path, topic, tokenizer):
        self.data = data
        self.persona = persona
        self.path = path
        self.topic = topic
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        # tokenizer weird behavior
        self.turn_ending = [628, 198]
        # tokenizer.encode("\n\n\n")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dial_tokens = []
        for i in range(len(self.data[index])):
            item1 = self.data[index][i]
            print(item1)
            sep = "\t"
            item2 = self.persona[index][i]
            print(item2)
            sep = "\t"  
            item3 = self.path[index][i]
            print(item3)
            sep = "\t"
            item4 = self.topic[index][i]
            print(item4)
            dial_tokens.append(tokenizer.encode(item1)+tokenizer.encode(sep)+tokenizer.encode(item2)+tokenizer.encode(sep)+tokenizer.encode(item3)+tokenizer.encode(sep)+tokenizer.encode(item4)+self.turn_ending)
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        return role_ids, dial_tokens
        
    def collate(self, unpacked_data):
        return unpacked_data

train_dataset = TopicShiftDataset(train_data, train_persona, train_path, train_topic, tokenizer)

val_dataset = TopicShiftDataset(val_data, val_persona, val_path, val_topic, tokenizer)

role_ids, dial_tokens = train_dataset.__getitem__(4)

batch_size = 1

train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=train_dataset.collate)

val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=train_dataset.collate)

"""### Load Model"""

# load the model
model_A = GPT2LMHeadModel.from_pretrained("gpt2", return_dict=False)
model_B = GPT2LMHeadModel.from_pretrained("gpt2", return_dict=False)


device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)

"""### Define the loss function"""

class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
                                       
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
    
    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        
        if reduce is "batch":
            # shape : scalar
            loss = loss.mean()

    return loss

criterion = SequenceCrossEntropyLoss()

"""### Training function"""

def train_one_iter(batch, update_count, fp16=False):
    role_ids, dialog_tokens = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
    # print(dial_inputs)
    past = None
    all_logits = []
    
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            # breakpoint()
            logits, past = model_A(dial_turn_inputs, past_key_values=past)
            all_logits.append(logits)
        else:
            # breakpoint()
            logits, past = model_B(dial_turn_inputs, past_key_values=past)
            all_logits.append(logits)
            
    if all_logits:
      all_logits = torch.cat(all_logits, dim=1)
      # target
      all_logits = all_logits[:, :-1].contiguous()
      target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
      target_mask = torch.ones_like(target).float()
      
      loss = criterion(all_logits, target, target_mask, label_smoothing=0.02, reduce="batch")   
      loss /= num_gradients_accumulation
  
      loss.backward()
          
      record_loss = loss.item() * num_gradients_accumulation
      perplexity = np.exp(record_loss)
      
      return record_loss, perplexity
    
    else:
      return None, None
    


def validate(dataloader):
    with torch.no_grad():
        pbar = progress_bar(dataloader)

        total_ppl = []

        for batch in pbar:
            
            if sum([len(item) for item in batch[0][1]]) > 1024:
                continue
            
            role_ids, dialog_tokens = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

            past = None
            all_logits = []

            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    logits, past = model_A(dial_turn_inputs, past_key_values=past)
                    all_logits.append(logits)
                else:
                    logits, past = model_B(dial_turn_inputs, past_key_values=past)
                    all_logits.append(logits)
            if all_logits:
              all_logits = torch.cat(all_logits, dim=1)
              # target
              all_logits = all_logits[:, :-1].contiguous()
              target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
              target_mask = torch.ones_like(target).float()
              
              loss = criterion(all_logits, target, target_mask, label_smoothing=-1, reduce="sentence")      
  
              ppl = torch.exp(loss)
              total_ppl.extend(ppl.tolist())

              print(f"Epcoh {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        
              return np.mean(total_ppl)

"""### Optimizer

We use the popular AdamW + Warmup
"""

# define hyper-parameters
num_epochs = 10
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
no_decay = ['bias', 'ln', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=100,
                                            num_training_steps=num_train_optimization_steps)

"""### Training - CONCATENATION"""

# os.makedirs("models", exist_ok=True)

update_count = 0
progress_bar = tqdm.tqdm_notebook
start = time.time()
old_ppl = -float('Inf')

for ep in range(num_epochs):

    "Training"
    pbar = progress_bar(train_dataloader)
    model_A.train()
    #model_B.train()
    
    for batch in pbar:
        batch = batch[0]
        
        # without relative position, we skip dialogs
        if sum([len(item) for item in batch[1]]) > 1024:
            continue
            
        record_loss, perplexity = train_one_iter(batch, update_count, fp16=False)
        
        update_count += 1

        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            # update for gradient accumulation
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end
            
            # show progress
            pbar.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)

    "Evaluation"
    model_A.eval()
    #model_B.eval()
    ppl = validate(val_dataloader)
    
    # save the model for later use
    torch.save([model_A.state_dict(), model_B.state_dict()], f"PATH_to_save_the_model_{ep}.pth")

