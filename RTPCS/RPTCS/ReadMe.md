# RPTCS: A Reinforced Persona-aware Topic-guiding Conversational System

A Reinforced Persona-aware Topic-guiding Conversational System <b>RPTCS</b>.

******Directory Tree of the SetUp:******

```
|___________/Codes
|	       |___ /MLCS
|	       		|___ MLCS.py				# script to train mle-loss based conversational system.
|
|	       |___ /RPTCS
|	       		|___ dataset.py				# script to load the custom dataset by creation of a pytorch Dataset class
|	       		|___ rlmain.py				# script to fine-tune mlcs with RL loss
|	       		|___ rlutils.py                     # script containing utility functions and reward functions for the RL fine-tuning task
|                       |___ ppo.py 				# script containing implementation of buffer memory
|                       |___ loss.py 				# script containing implementation of Sequence Cross Entropy Loss
|              		|___ rltest.py 				# script to interact with the fine-tuned model.
|
|
|	       |___ /NLI
|                       |___ load_bert.py 			# script to train the nli model
|                       |___ persona_nli.py 			# script to implement nli model
|
|
|
|
|___________/Data	 				 	         
|              |___ PTCD_train.csv					# Persona aware topic guiding conversational training dataset
|              |___ PTCD_val.csv					# Persona aware topic guiding conversational validation dataset
|              |___ PTCD_test.csv					# Persona aware topic guiding conversational test dataset 
```

******REQUIREMENTS******
1. numpy: version '1.21.2'
2. pandas: version '1.3.4'
3. transformers: version '4.11.2'
4. tqdm: version: version '4.62.3'
5. torch: version '1.10.0'


******FINE-TUNING RL MODEL******

1. Provide all the arguments in the "rlmain.py" file.
2. Go to terminal window and enter to start RL fine-tuning:<br>
```python
python rlmain.py 
```

******Arguments in rlmain.py:******

modelname:str, 'the desired modelname', <br>
csvfile:str, the csv file to load the annotated dataset from <br>
device:str, Default='cuda' <br>
n_epochs:int, Default=1 <br>
batch_size:int, Default=1 <br>
mini_batch=int, Default=1 <br>
train_single_model:bool, Whether to fine-tune both agent and user or either one of them during RL fine tuning, Default=True <br>
single_model_to_train:str, Which model of train 'agent' or 'user', Default:'agent', <br>
num_candidates:int, number of candidates to generate at a turn for the agent, Default=3 <br>
recompute_log_prob:bool, Whether to recompute the log probability of the generated candidates, Default= True <br>
average_sent_loss:bool, Whether to average the loss the over the entire sentence for the generated candidates, Default=True <br>
max_candidate_length:int, Maximum length of generated candidates, Default=50 <br>
human_reward:int, Default=10 <br>
beta2:float, Default=2 <br>
beta3:float, Default=2 <br>
beta4:float, Default=2 <br>
top_p:float, The probability sum threshold to consider when generating tokens for the candidates,  Default=0.9 <br>
temperature:float, The temprate value when calculating the loss, Default=0.8 <br>
use_recent_past:bool, Whether to consider the recent past <br>
warmup_steps:int, number of warm up step to be given to the scheduler, Default=10 <br>
print_every:int, number of steps before printing the loss Default=1 <br>
evaluate_every:int, Iterations before evaluation, Default=1 <br>
learning_rate:float, Default=2e-05 <br>
epsilon:float, Default=0.2 <br>
loadModel:bool, Whether to load the pretrained language model for fine-tuning, Default=True <br>
loadFilename:str, path to the saved pretrained language model <br>
pad_token_id:int, Default=2 <br>
seedvalue:int, Default=10 <br>
use_jaccard:bool, Whether to use non-repetitiveness as reward, Default=True <br>
use_cosine:bool, Whether to use utterance_topic_consistency as a reward, Default=True <br>
use_per_utt:bool, Whether to use utterance_persona_consistency as a reward, Default=True <br>
use_context:bool, Whether to use context_consistency as a reward, Default=True <br>
gamma1:float, weight for the non-repetitiveness reward, Default=0.2 <br>
gamma2:float, weight for the utterance_topic_consistency reward, Default=0.3 <br>
gamma3:float, weight for the utterance_persona_consistency reward, Default=0.3 <br>
gamma4:float, weight for the context_consistency reward, Default=0.2 <br>
