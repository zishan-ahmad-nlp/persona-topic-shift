import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='GPU_NUMBER'

import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
from nltk.translate import meteor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import os
import pdb
import spacy
import pandas as pd
import json
import tqdm
import datetime
from tqdm.notebook import tqdm_notebook
from nltk import word_tokenize
import random
import pdb
from rlutils import collect_samples, ppo_step, generate_n_candidates, convert_sentences_to_strings, expand_inputs_for_N_candidates
from torch.utils.data import DataLoader, Dataset
from loss import SequenceCrossEntropyLoss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ppo import PPOMemory
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup#, RobertaForSequenceClassification, RobertaTokenizer
# from simpletransformers.classification import ClassificationModel,    
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
from dataset import TopicShiftDataset
from persona_nli import *
import warnings
warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')

class Trainer():

    def __init__(self,
                 modelname,
                 csvfile,
                 n_epochs,
                 print_every,
                 learning_rate,
                 epsilon,
                 human_reward,
                 average_sent_loss,
                 device,
                 num_candidates,
                 max_candidate_length,
                 top_p,
                 warmup_steps,
                 pad_token_id,
                 evaluate_every,
                 use_jaccard,
                 use_cosine,
                 use_per_utt,
                 use_context,
                 mini_batch,
                 temperature,
                 use_recent_past,
                 recompute_log_prob,
                 gamma1,
                 gamma2,
                 gamma3,
                 gamma4,
                 train_single_model=False,
                 single_model_to_train=None,
                 loadModel=False,
                 batch_size=None,
                 loadFilename=None,
                 seedvalue=10):

        self.seedvalue = seedvalue
        self.train_single_model = train_single_model
        self.single_model_to_train = single_model_to_train
        self.nlp = spacy.load("en_core_web_sm")
        self.human_reward = human_reward
        self.seed(seedvalue)
        self.use_recent_past = use_recent_past
        self.temperature=temperature
        self.use_jacc = use_jaccard
        self.use_cosine = use_cosine
        self.use_context = use_context
        self.use_per_utt = use_per_utt

        self.average_sent_loss = average_sent_loss
        self.mini_batch = mini_batch
        self.evaluate_every = evaluate_every
        self.csvfile = csvfile
        self.modelname = modelname
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.recompute_log_prob = recompute_log_prob
        self.num_candidates = num_candidates
        self.pad_token_id = pad_token_id
        self.max_candidate_length = max_candidate_length
        
        
        self.top_p = top_p
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

        self.device = device
        
        self.loadModel = loadModel
        self.loadFilename = loadFilename
        self.make_model_save_dir()
        self.make_stats_dir()
        

        self.getDataset()
        
        self.initialize_models()
        self.configure_optimizer()
        
        self.buffer_memory = PPOMemory()
        
        self.saveModelConfig()
        self.criterion = SequenceCrossEntropyLoss()

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4



    def saveModelConfig(self):
        if self.train_single_model:
            config_model_train = self.single_model_to_train
            print('Training Only :', self.single_model_to_train)
        else:
            config_model_train = 'Both Models being Trained.'
            print('Both Models being Trained.')
        config = {'Basic Info': [datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")],
                  'NOTES': 'GPT2-MEDIUM',
                  'modelname': self.modelname,
                  'Training only one Model': self.train_single_model,
                  'Training Models': config_model_train,
                  'device': self.device,
                  'use_jaccard_similarity': self.use_jacc,
                  'use_cosine': self.use_cosine,
                  'use_per_utt': self.use_per_utt,
                  'use_context' : self.use_context,
                  'modelLoaded': self.loadFilename,
                  'human_reward': self.human_reward,
                  'average_sent_loss' : self.average_sent_loss,
                  'n_epochs': self.n_epochs,
                  'use_recent_past': self.use_recent_past,
                  'temperature': self.temperature,
                  'learning_rate': self.learning_rate,
                  'epsilon': self.epsilon,
                  'num_candidates': self.num_candidates,
                  'pad_token_id': self.pad_token_id,
                  'max_candidate_length': self.max_candidate_length,
                  'recompute_log_prob': self.recompute_log_prob,
                  'evaluate_every': self.evaluate_every,
                  'top_p': self.top_p,
                  'warmup_steps': self.warmup_steps,
                  'batch_size':self.batch_size,
                  'seed': self.seedvalue}
        configfilename = os.path.join(self.savefolder, self.modelname, 'config')
        if not os.path.exists(configfilename):
            os.makedirs(configfilename)
        configfilename = configfilename + '/config' + '_' + self.modelname + '.json'
        with open(configfilename, 'w') as f:
            json.dump(config, f)

    def seed(self,seed=10):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def extract_data(self, csvfile):
        df = pd.read_csv(csvfile)

        data = []
        persona = []
        path = []
        topic =[]
        for i in tqdm.trange(len(df)):
            if df['speaker'][i] == 0:
                text = "A:" + str(df["utterance"][i])
                persona.append(str(df["persona"][i]))
                path.append(str(df["path"][i]))
                topic.append(str(df["topic"][i]))
                data.append(text)
            else:
                text = "B:" + str(df["utterance"][i])
                persona.append(str(df["persona"][i]))
                path.append(str(df["path"][i]))
                topic.append(str(df["topic"][i]))
                data.append(text)
        return data, persona, path, topic


        
    def utteranceToConversation(self, csvfile, data, persona, path, topic):
        df = pd.read_csv(self.csvfile)
        values=df['conv_id'].unique().tolist()
        conv_ids = df['conv_id'].tolist()

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
        return dataset, personaset, pathset, topicset

  
          
    def convertDicttoList(self, data: dict):
        return list(data.values())

    def random_split_data(self, data, persona, path, topic):
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
        return train_data, val_data, train_persona, val_persona, train_path, val_path, train_topic, val_topic


    def getDataset(self):
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        data, persona, path, topic = self.extract_data(self.csvfile)
        data, persona, path, topic = self.utteranceToConversation(self.csvfile, data, persona, path, topic)


        #self.traindata, self.valdata, self.traintopic, self.valtopic = train_data, val_data, train_topic, val_topic
        self.traindata, self.valdata, self.trainpersona, self.valpersona, self.trainpath, self.valpath, self.traintopic, self.valtopic = self.random_split_data(data, persona, path, topic)


        
        traindata_ = TopicShiftDataset(self.traindata,
                                     self.trainpersona,
                                     self.trainpath,
                                     self.traintopic,
                                     self.tokenizer)

        
        self.turn_ending = traindata_.get_turn_ending()
        
        valdata_ = TopicShiftDataset(self.valdata,
                                   self.valpersona,
                                   self.valpath,
                                   self.valtopic,
                                   self.tokenizer)
        

        self.train_dataloader = DataLoader(dataset=traindata_,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=traindata_.collate)
        
        self.val_dataloader = DataLoader(dataset=valdata_,
                                         shuffle=False,
                                         batch_size=self.batch_size,
                                         collate_fn=valdata_.collate)

    def initialize_models(self):
        if not self.train_single_model:
            self.model_A = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_B = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            if self.single_model_to_train == 'agent':
                self.model_A = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2")
            else:
                self._model_B = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2")

        if self.loadModel:
            if self.loadFilename:
                model_A_state_dict, model_B_state_dict = torch.load(self.loadFilename, map_location=self.device)
                if not self.train_single_model:
                    self.model_A.load_state_dict(model_A_state_dict)
                    self.model_A_ref.load_state_dict(model_A_state_dict)
                    self.model_B.load_state_dict(model_B_state_dict)
                    self.model_B_ref.load_state_dict(model_B_state_dict)
                    self.model_A = self.model_A.to(self.device)
                    self.model_A_ref = self.model_A_ref.to(self.device)
                    self.model_B = self.model_B.to(self.device)
                    self.model_B_ref = self.model_B_ref.to(self.device)
                    self.model_A.train()
                    self.model_B.train()
                    self.model_A_ref.eval()
                    self.model_B_ref.eval()
                else:
                    if self.single_model_to_train == 'agent':
                        self.model_A.load_state_dict(model_A_state_dict)
                        self.model_A_ref.load_state_dict(model_A_state_dict)
                        self.model_A = self.model_A.to(self.device)
                        self.model_A_ref = self.model_A_ref.to(self.device)
                        self.model_A.train()
                        self.model_A_ref.eval()
                        #self.model_B.load_state_dict(model_B_state_dict) 
                        #self.model_B = self.model_B.to('cuda')
                        #self.model_B.eval()
                        self.model_B = None
                        self.model_B_ref = None
                    else:
                        self.model_B.load_state_dict(model_B_state_dict)
                        self.model_B_ref.load_state_dict(model_B_state_dict)
                        self.model_B = self.model_B.to(self.device)
                        self.model_B_ref = self.model_B_ref.to(self.device)
                        self.model_B.train()
                        self.model_B_ref.eval()
                        self.model_A = None
                        self.model_A_ref = None
                print('\n')
                print("Models loaded from file ", self.loadFilename)
            else:
                print('Models not loaded since directory not provided.')
        print(f"Models Initalized!")
        print('\n')


    def configure_optimizer(self):
        
        self.num_train_optimization_steps = self.n_epochs * len(self.traindata) # // self.batch_size

        if not self.train_single_model:
            param_optimizer = list(self.model_A.named_parameters()) + list(self.model_B.named_parameters())
        else:
            if self.single_model_to_train == 'agent':
                param_optimizer = list(self.model_A.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters,
                                           lr=self.learning_rate,
                                           eps=1e-06)

        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        #                                                 num_warmup_steps=self.warmup_steps,
        #                                                 num_training_steps=self.num_train_optimization_steps)

        '''self.scheduler = WarmupLinearSchedule(self.optimizer,
                                                 warmup_steps=self.warmup_steps,
                                                 t_total=self.num_train_optimization_steps)'''


    def get_candidate_lengths(self, candidates):



        avg_iter_length = []
        
        for i in candidates:
            candidate_sentence = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            avg_iter_length.append(len(candidate_sentence.split()))
        return avg_iter_length

    def get_up_con_score(self, candidates, turn_num, dial_inputs):

        up_con_score = []
        for i in candidates:
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            if(turn_num>=2):
                persona = self.tokenizer.decode(dial_inputs[turn_num-1].tolist()[0]).split('\n')[0].split('\t')[1].strip()
            else:
                persona = ''

            turn = []
            turn.append(persona)
            s = get_nli(candidate, turn)
            z = torch.argmax(s)
            if z==0:
                up_con_score.append(-1)
            elif z==2:
                up_con_score.append(1)
            else:
                up_con_score.append(0)

        return up_con_score


    def get_meteor_score(self, candidates, current_sentence):

        meteor_score_list = []
        
        for i in candidates:
            reference = []
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            predicted = word_tokenize(candidate) 
            ref = word_tokenize(current_sentence)
            reference.append(ref)
            meteor_score = round(meteor(reference, predicted),2)  
            meteor_score_list.append(meteor_score)         
        return meteor_score_list 


    def get_utt_t_score(self, candidates, turn_num, dial_inputs):

        utt_t_list = []
        
        for i in candidates:
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            if(turn_num>=2):
                topic = self.tokenizer.decode(dial_inputs[turn_num-1].tolist()[0]).split('\n')[0].split('\t')[3].strip()
            else:
                topic = ''
            turn = []
            turn.append(candidate)
            turn.append(topic)
            turn=model.encode(turn)
            score = cosine_similarity([turn[0]], turn[1:])[0][0]
            utt_t_list.append(score)
        
        return utt_t_list



    def validate_model(self, dataloader):

        with torch.no_grad():
            if not self.train_single_model:
                self.model_A.eval()
                self.model_B.eval()
            else:
                if self.single_model_to_train == 'agent':
                    self.model_A.eval()
                else:
                    self.model_B.eval()

            with torch.no_grad():
                
                progress_bar = tqdm_notebook
                pbar = progress_bar(dataloader)
               
                total_ppl = []
                total_loss = []
                total_r_len = []
                total_meteor = []
                total_utt_t_cons = []
                total_up_cons = []

                for batch in pbar:

                    if sum([len(item) for item in batch[0][1]]) > 1024:
                        continue

                    role_ids, dialog_tokens = batch[0]


                    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
                    past = None
                    past_ = None
                    all_logits = []
                    target = []

                    for turn_num, dial_turn_inputs in enumerate(dial_inputs):

                        if not self.train_single_model:
                            if role_ids[turn_num] == 0:
                                outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                            else:
                                outputs = self.model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                        else:
                            if self.single_model_to_train == 'agent':
                                if role_ids[turn_num] == 0:
                                    dial_turn_inputs = dial_turn_inputs.to(self.device)
                                    outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                    past = outputs[1]
                                    all_logits.append(outputs[0])
                                    target.append(dial_turn_inputs)
                                    generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          torch.tensor(self.tokenizer.encode("A:")).unsqueeze(0).to('cuda'),
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                    output = self.model_A(expand_inputs_for_N_candidates(dial_turn_inputs,
                                                                                         self.num_candidates),
                                                                                         past_,
                                                                                         return_dict=False)
                                    past_ = output[1]
                                    current_sentence = self.tokenizer.decode(dial_turn_inputs.tolist()[0][2:]).split('\t')[0]

                    all_logits = torch.cat(all_logits, dim=1)
                    all_logits = all_logits[:, :-1].contiguous()

                    if not self.train_single_model:
                        target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
                    else:
                        target = torch.cat(target, dim=1)[:, 1:].contiguous()
                    
                    target_mask = torch.ones_like(target).float()

                    loss = self.criterion(all_logits, target, target_mask, label_smoothing=-1, reduce='sentence')
                    total_loss.extend(loss.tolist())

                    ppl = torch.exp(loss)
                    total_ppl.extend(ppl.tolist())
                    


                    average_lengths = self.get_candidate_lengths(generated_sequence)
                    total_r_len.append(np.mean(average_lengths))

                    meteor_scores = self.get_meteor_score(generated_sequence, current_sentence)
                    total_meteor.append(np.mean(meteor_scores))

                    up_cons_scores = self.get_up_con_score(generated_sequence, turn_num, dial_inputs)
                    total_up_cons.append(np.mean(up_cons_scores))

                    utt_t_cons_scores = self.get_utt_t_score(generated_sequence, turn_num, dial_inputs)
                    total_utt_t_cons.append(np.mean(utt_t_cons_scores))

                print('\n')
                print(f"Validation Perplexity: {np.mean(total_ppl)}")

                # average_lengths = self.get_candidate_lengths(generated_sequence)
                print(f"Overall Average candidate length: {np.mean(total_r_len)}")
                print(f"Overall Meteor score: {np.mean(total_meteor)}")
                print(f"Overall utterance topic const score: {np.mean(total_utt_t_cons)}")
                print(f"Overall utterance persona const score: {np.mean(total_up_cons)}")
                

        return np.mean(total_ppl), np.mean(total_loss), np.mean(average_lengths)
    

    def make_stats_dir(self):
        
        self.statsfolder = os.path.join(os.getcwd(), self.savefolder, self.modelname, 'stats')
        if not os.path.exists(self.statsfolder):
            os.makedirs(self.statsfolder)


    def make_model_save_dir(self):
        
        self.savefolder = os.path.join(os.getcwd(), 'Path_to_save_the_trained_model')
        if not os.path.exists(self.savefolder):
            print("Model save folder doesn't exist.")
            os.makedirs(self.savefolder)
            print(f"Created folder {self.savefolder} to save the models.")


    def save_models(self, num_iter):
        
        modeldir = os.path.join(self.savefolder, self.modelname)
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
            print('Created Directory for saving models!')
        filename = modeldir + '/' + self.modelname + '_' + str(num_iter) + ".pth"
        #torch.save([self.model_A.state_dict(), self.model_B.state_dict()], filename)
        torch.save(self.model_A.state_dict(), filename)

    def modified_train_one_iter(self, batch):
        dial_inputs, role_ids, scores_dict = collect_samples(batch,
                                                             model_A=self.model_A_ref,
                                                             model_B=self.model_B,
                                                             top_p=self.top_p,
                                                             eos_token_id=self.turn_ending[0],
                                                             pad_token_id=self.turn_ending[1],
                                                             average_sent_loss=self.average_sent_loss,
                                                             max_gen_length=self.max_candidate_length,
                                                             buffer_memory=self.buffer_memory,
                                                             use_cosine=self.use_cosine,
                                                             use_per_utt=self.use_per_utt,
                                                             use_context=self.use_context,
                                                             device=self.device,
                                                             num_candidates=self.num_candidates,
                                                             human_reward=self.human_reward,
                                                             use_jacc=self.use_jacc,
                                                             tokenizer=self.tokenizer,
                                                             criterion=self.criterion,
                                                             temperature=self.temperature,
                                                             use_recent_past=self.use_recent_past,
                                                             recompute_log_prob=self.recompute_log_prob,
                                                             nlp=self.nlp,
                                                             train_single_model=self.train_single_model,
                                                             model_to_train=self.single_model_to_train,
                                                             gamma1=self.gamma1,
                                                             gamma2=self.gamma2,
                                                             gamma3=self.gamma3,
                                                             gamma4=self.gamma4)

        log_dict = ppo_step(model_A=self.model_A,
                            model_B=self.model_B,
                            buffer_memory=self.buffer_memory,
                            train_single_model=self.train_single_model,
                            dial_inputs= dial_inputs,
                            model_to_train=self.single_model_to_train,
                            device=self.device,
                            ppo_epsilon=self.epsilon,
                            num_candidates=self.num_candidates,
                            use_recent_past=self.use_recent_past,
                            average_sent_loss=self.average_sent_loss,
                            criterion=self.criterion,
                            optimizer=self.optimizer,
                            role_ids=role_ids)

        self.buffer_memory.clear_memory()

        return log_dict, scores_dict 
 
    def train(self):

        update_count = 0
        progress_bar = tqdm_notebook

        val_ppl = []
        val_loss = []

        rewards = []
        kl = []
        clip_frac = []

        cos_sim_scores = []
        per_utt_scores = []
        jacc_scores = []
        context_adequacy_scores = []
        

        best_ppl = None
        
        length = None
        
        iters = None
        
        #strategies = None

        pbar = progress_bar(self.train_dataloader)

        for i in range(self.n_epochs):
            if not self.train_single_model:
                self.model_A.train()
                self.model_B.train()
            else:
                if self.single_model_to_train == 'agent':
                    self.model_A.train()
            for batch in pbar:
                if sum([len(item) for item in batch[0][1]]) > 1024 - self.max_candidate_length:
                    continue

                print(f"ITERATION: {update_count}")

                batch = batch[0]
                log_dict, scores_dict  = self.modified_train_one_iter(batch)

                clip_frac.append(log_dict['clip_frac'])
                kl.append(log_dict['approx_kl'])
                rewards.append(log_dict['reward'])

                cos_sim_scores.extend(scores_dict['cos_sim_scores'])
                per_utt_scores.extend(scores_dict['per_utt_scores'])
                jacc_scores.extend(scores_dict['jacc_scores'])
                context_adequacy_scores.extend(scores_dict['context_adequacy_scores'])
                

                np.save(self.statsfolder + '/' + 'cos_sim_scores.npy', np.array(cos_sim_scores))
                np.save(self.statsfolder + '/' + 'per_utt_scores.npy', np.array(per_utt_scores))
                np.save(self.statsfolder + '/' + 'context_adequacy_scores.npy', np.array(context_adequacy_scores))
                np.save(self.statsfolder + '/' + 'non-repetitiveness_scores.npy', np.array(jacc_scores))
                
                update_count += 1


                print('update count is:', update_count)

                if  update_count % self.evaluate_every == 0:
                    
                    ppl, loss, average_length = self.validate_model(self.val_dataloader)
                    
                    if best_ppl is None:

                        best_ppl = ppl
                        iters = update_count
                        
                        length = average_length
                        
                        if update_count > 20 and update_count < 22:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                        
                    else:
                        if ppl < best_ppl:
                            best_ppl = ppl
                            iters = update_count
                            
                            length = average_length
                            
                        if update_count > 20 and update_count < 22:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                
                    print('\n')
                    print(f'Best Perplexity Found so far {best_ppl} for iteration: {iters}')
                    print('\n')
                    
                    val_ppl.append(ppl)
                    val_loss.append(loss)
                    
                                
                    np.save(self.statsfolder + '/' + 'val_PPL_iter'  + '.npy', np.array(val_ppl))
                    
                    
                    np.save(self.statsfolder + '/' + 'train_rewards' + '.npy', np.array(rewards))
                    np.save(self.statsfolder + '/' + 'train_kl' + '.npy', np.array(kl))
                    np.save(self.statsfolder + '/' + 'train_clip_frac' + '.npy', np.array(clip_frac))
                    np.save(self.statsfolder + '/' + 'best_ppl_iteration_value' + '.npy', np.array(iters))
                    


                    #self.initialize_strategy_count()
    
                    if not self.train_single_model:
                        self.model_A.train()
                        self.model_B.train()
                    else:
                        if self.single_model_to_train == 'agent':
                            self.model_A.train()
                #if update_count == 17:
                #    return best_ppl, iters
        return best_ppl, iters

if __name__ == '__main__':
    trainer = Trainer(modelname='PATH_TO_SAVE_THE_MODEL',
                      csvfile="DATASET_PATH",
                      device='cuda',
                      n_epochs=1,
                      batch_size=1,
                      mini_batch=20,
                      train_single_model=True,
                      single_model_to_train = 'agent',
                      num_candidates=3,
                      recompute_log_prob=True,
                      average_sent_loss=True,
                      max_candidate_length=50,
                      human_reward=10,
                      top_p=0.9,
                      temperature=0.8,
                      use_recent_past=True,
                      warmup_steps=10,
                      print_every=1,
                      evaluate_every=1,
                      learning_rate=2e-5,
                      epsilon=0.2,
                      loadModel=True,
                      loadFilename="PATH_OF_MLCS_TRAINED",
                      pad_token_id=2,
                      seedvalue=10, # 10 should be the seed value since pre trained on the same seed. 
                      use_jaccard=True,
                      use_cosine=True,
                      use_per_utt=True,
                      use_context=True,
                      gamma1=0.2,
                      gamma2=0.3,
                      gamma3=0.3,
                      gamma4=0.2)

    trainer.train()