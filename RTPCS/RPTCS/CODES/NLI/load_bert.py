import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch import nn

logger = logging.getLogger(__name__)

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding='utf-8') as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

class PersonanliProcessor(DataProcessor):
	"""Processor for the Personanli data set (GLUE version)."""

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "path_to_train_nli_dataset")), "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "path_to_test_nli_dataset")),
			"test")

	def get_labels(self):
		"""See base class."""
		return ["contradiction", "entailment", "neutral"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, line[0])
			text_a = line[1]
			text_b = line[2]
			label = line[3]
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples
		
	def create_batch(self, turn, persona, set_type="predict"):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(zip(turn, persona)):
			guid = "%s-%s" % (set_type, i)
			text_a = line[0]
			text_b = line[1]
			label = "entailment"
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {label : i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids: 0   0   0   0  0     0 0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambigiously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		label_id = label_map[example.label]
		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_id=label_id))
	return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()

def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x

class bert_model(object):
	def __init__(self):


		# args = parser.parse_args()
		# self.args = args
		self.max_seq_length = 128
		self.eval_batch_size = 32
		self.device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
		self.processor = PersonanliProcessor()
		self.num_labels = 3
		self.label_list = self.processor.get_labels()

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

		# Load a trained model that you have fine-tuned
		# output_model_file = os.path.join("nli_model/", "pytorch_model.bin")
		output_model_file = "Path_of_fine-tuned_model"
		model_state_dict = torch.load(output_model_file)
		model = BertForSequenceClassification.from_pretrained('bert-base-uncased', state_dict=model_state_dict, num_labels=self.num_labels)
		model.to(self.device)
		self.model = model

	def predict_label(self, turn, personas_items):
		eval_examples = self.processor.create_batch(turn, personas_items)
		eval_features = convert_examples_to_features(eval_examples, self.label_list, self.max_seq_length, self.tokenizer)
		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(eval_examples))
		logger.info("  Batch size = %d", self.eval_batch_size)
		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
		eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
		# # Run prediction for full data
		eval_sampler = SequentialSampler(eval_data)
		eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

		self.model.eval()
		# softmax = nn.Softmax(dim=-1)
		# eval_loss, eval_accuracy = 0, 0
		# nb_eval_steps, nb_eval_examples = 0, 0
		mapper = {int("0"):"contradiction", int("1"):"entailment", int("2"):"neutral"}
		for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
			input_ids = input_ids.to(self.device)
			input_mask = input_mask.to(self.device)
			segment_ids = segment_ids.to(self.device)
			label_ids = label_ids.to(self.device)

			with torch.no_grad():
				tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
				logits = self.model(input_ids, segment_ids, input_mask)

			logits = logits.detach().cpu().numpy()
			idx_max = np.argmax(logits, axis=1)
			val_max = np.max(logits, axis=1)
			score = 0
			for idx_p, val_p in enumerate(idx_max):
				if(mapper[val_p]=="entailment"):
					score += 1
				if(mapper[val_p]=="contradiction"):
					score -= 1
			return score, logits


