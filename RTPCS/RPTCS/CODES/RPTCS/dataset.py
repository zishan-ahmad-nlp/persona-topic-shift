import torch
from torch.utils.data import DataLoader, Dataset
import pdb

class TopicShiftDataset(Dataset):
    def __init__(self, data, persona, path, topic, tokenizer):
        self.data = data
        self.persona = persona
        self.path = path
        self.topic = topic
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = [628, 198]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):         

        dial_tokens = []

        for i in range(len(self.data[index])):
            item1 = self.data[index][i]
            sep = "\t"
            item2 = self.persona[index][i]
            sep = "\t"  
            item3 = self.path[index][i]
            sep = "\t"
            item4 = self.topic[index][i]


            dial_tokens.append(self.tokenizer.encode(item1)+self.tokenizer.encode(sep)+self.tokenizer.encode(item2)+self.tokenizer.encode(sep)+self.tokenizer.encode(item3)+self.tokenizer.encode(sep)+self.tokenizer.encode(item4)+self.turn_ending)
            


        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        # print(role_ids)
        return role_ids, dial_tokens

    def collate(self, unpacked_data):
        return unpacked_data

    def get_turn_ending(self):
        return self.turn_ending
