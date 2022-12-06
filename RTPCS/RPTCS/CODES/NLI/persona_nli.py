from load_bert import bert_model
import torch
from torch import nn


bert = bert_model()
softmax = nn.Softmax(dim=-1)

def get_nli(uttr, persona):
	x, logits = bert.predict_label([uttr]*len(persona), persona) 
	logits = torch.tensor(logits)
	ent_p = softmax(logits)
	ent_p = torch.sum(ent_p,0)/ent_p.size(0)
	return ent_p
