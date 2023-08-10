import os.path

import torch
import torch.nn as nn

from pandas import DataFrame
import pandas as pd

from model.utilities import *
from model.ModelZoo import ModelZoo
from model.Logger import Logger
from model.Timer import Timer

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()

# Dataset & Model
parser.add_argument("-d", 		dest = "dataset", 	type = str, default = "musical_instruments", 	help = "Dataset for Running Experiments (Default: musical_instruments)")
parser.add_argument("-m", 		dest = "model", 	type = str, default = "ANR", 					help = "Model Name, e.g. DeepCoNN|DAttn|ANR|ANRS (Default: ANR)")

# General Hyperparameters
parser.add_argument("-bs", 			dest = "batch_size", 	type = int, 	default = 128,	 		help = "Batch Size (Default: 128)")
parser.add_argument("-e", 			dest = "epochs", 		type = int, 	default = 25, 			help = "Number of Training Epochs (Default: 25)")
parser.add_argument("-lr", 			dest = "learning_rate", type = float, 	default = 2E-3, 		help = "Learning Rate (Default: 0.002, i.e 2E-3)")
parser.add_argument("-opt", 		dest = "optimizer", 	type = str, 	default = "Adam", 		help = "Optimizer, e.g. Adam|RMSProp|SGD (Default: Adam)")
parser.add_argument("-loss_func", 	dest = "loss_function", type = str, 	default = "MSELoss", 	help = "Loss Function, e.g. MSELoss|L1Loss (Default: MSELoss)")
parser.add_argument("-dr",			dest = "dropout_rate", 	type = float, 	default = 0.5, 			help = "Dropout rate (Default: 0.5)")

# Dataset-Specific Settings (Document Length, Vocabulary Size, Dimensionality of the Embedding Layer, Source of Pretrained Word Embeddings)
parser.add_argument("-MDL", 		dest = "max_doc_len", 		type = int, 	default = 500, 		help = "Maximum User/Item Document Length (Default: 500)")
parser.add_argument("-v", 			dest = "vocab_size", 		type = int, 	default = 50000, 	help = "Vocabulary Size (Default: 50000)")
parser.add_argument("-WED", 		dest = "word_embed_dim", 	type = int, 	default = 300, 		help = "Number of Dimensions for the Word Embeddings (Default: 300)")
parser.add_argument("-p", 			dest = "pretrained_src", 	type = int, 	default = 1,		help = "Source of Pretrained Word Embeddings? \
	0: Randomly Initialized (Random Uniform Dist. from [-0.01, 0.01]), 1: w2v (Google News, 300d), 2: GloVe (6B, 400K, 100d) (Default: 1)")

# ANR Hyperparameters
parser.add_argument("-K", 		dest = "num_aspects", 	type = int, 	default = 5, 	help = "Number of Aspects (Default: 5)")
parser.add_argument("-h1", 		dest = "h1", 			type = int, 	default = 10, 	help = "Dimensionality of the Aspect-level Representations (Default: 10)")
parser.add_argument("-c", 		dest = "ctx_win_size", 	type = int, 	default = 3, 	help = "Window Size (i.e. Number of Words) for Calculating Attention (Default: 3)")
parser.add_argument("-h2", 		dest = "h2", 			type = int, 	default = 50, 	help = "Dimensionality of the Hidden Layers used for Aspect Importance Estimation (Default: 50)")
parser.add_argument("-L2_reg", 	dest = "L2_reg", 		type = float, 	default = 1E-6, help = "L2 Regularization for User & Item Bias (Default: 1E-6)")

# ANR Pretraining
parser.add_argument("-ARL_path", 	dest = "ARL_path", 	type = str, 	default = "", 	help = "Specify the file name for loading pretrained ARL weights! (Default: "", i.e. Disabled)")
parser.add_argument("-ARL_lr", 		dest = "ARL_lr", 	type = float, 	default = 0.01,	help = "RATIO of LR for fine-tuning the pretrained ARL weights (Default: 0.01)")


# Miscellaneous
parser.add_argument("-rs", 	dest = "random_seed", 			type = int, default = 1337, help = "Random Seed (Default: 1337)")
parser.add_argument("-dc", 	dest = "disable_cuda", 			type = int, default = 0, 	help = "Disable CUDA? (Default: 0, i.e. run using GPU (if available))")
parser.add_argument("-gpu", dest = "gpu", 					type = int, default = 0, 	help = "Which GPU to use? (Default: 0)")
parser.add_argument("-vb", 	dest = "verbose", 				type = int, default = 0, 	help = "Show debugging/miscellaneous information? (Default: 0, i.e. Disabled)")
parser.add_argument("-die", dest = "disable_initial_eval", 	type = int, default = 0, 	help = "Disable initial Dev/Test evaluation? (Default: 0, i.e. Disabled)")
parser.add_argument("-sm", 	dest = "save_model", 			type = str, default = "", 	help = "Specify the file name for saving model! (Default: "", i.e. Disabled)")

args = parser.parse_args()


# Check for availability of CUDA and execute on GPU if possible
args.use_cuda = not args.disable_cuda and torch.cuda.is_available()
del args.disable_cuda


# Initial Setup
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

if(args.use_cuda):
	select_gpu(args.gpu)
	torch.cuda.set_device(args.gpu)
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed(args.random_seed)
else:
	print("\n[args.use_cuda: {}] The program will be executed on the CPU!!".format( args.use_cuda ))


# Timer & Logging
timer = Timer()
timer.startTimer()

uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
args.input_dir = "./datasets/{}/".format( args.dataset )
args.out_dir = "./experimental_results/{} - {}/".format( args.dataset, args.model )
log_path = "{}{}-{}".format(args.out_dir, uuid, 'logs.txt')
logger = Logger(args.out_dir, log_path, args)


saved_models_dir = "./__saved_models__/{} - {}/".format( args.dataset, args.model )
model_path = "{}{}_{}.pth".format( saved_models_dir, args.save_model.strip(), args.random_seed )
# # Optional: Saving Model
# if(args.save_model != ""):
# 	saved_models_dir = "./__saved_models__/{} - {}/".format( args.dataset, args.model )
# 	mkdir_p(saved_models_dir)
# 	model_path = "{}{}_{}.pth".format( saved_models_dir, args.save_model.strip(), args.random_seed )

# Create model
mdlZoo = ModelZoo(logger, args, timer)
mdl = mdlZoo.createAndInitModel()

# Load weights
model_states = torch.load(model_path)

# Update Current Model, using the pretrained ARL weights
DESIRED_KEYS = ["shared_ANR_ARL.aspProj", "shared_ANR_ARL.aspEmbed.weight"]

pretrained_mdl_state_dict = model_states["mdl"]
pretrained_mdl_state_dict = {k: v for k, v in pretrained_mdl_state_dict.items() if k in DESIRED_KEYS}
print("\nLoaded pretrained model states:\n")
for pretrained_key in pretrained_mdl_state_dict.keys():
	print("\t{}".format( pretrained_key ))
current_mdl_dict = mdl.state_dict()
current_mdl_dict.update(pretrained_mdl_state_dict)
mdl.load_state_dict(current_mdl_dict)
print("\nPretrained model states transferred to current model! {}".format(model_path))

# Load training/validation/testing sets
train_set, train_loader, dev_set, dev_loader, test_set, test_loader = loadTrainDevTest(logger, args)
logger.log("Train/Dev/Test splits loaded! {}".format( timer.getElapsedTimeStr("init", conv2Mins = True) ))

# For evaluation
def evaluate(mdl, set_loader, epoch_num = -1, use_cuda = True, phase = "Dev", print_txt = True):

	all_rating_true = []
	all_rating_pred = []

	rating_pred_list = []  # list of tensors
	batch_rating_list = []   # list of tensors
	for batch_num, (batch_uid, batch_iid, batch_rating) in enumerate(set_loader):
		# Set to evaluation mode, important for dropout & batch normalization!
		mdl.eval()
		b_uid = batch_uid
		b_iid = batch_iid

		batch_uid = to_var(batch_uid, use_cuda = use_cuda, phase = phase)
		batch_iid = to_var(batch_iid, use_cuda = use_cuda, phase = phase)

		rating_pred = torch.squeeze(mdl(batch_uid, batch_iid))

		all_rating_true.extend(batch_rating)
		all_rating_pred.extend(rating_pred.data)

		rating_pred_list.append(rating_pred.data.numpy())   # .numpy() converts tensors to arrays
		batch_rating_list.append(batch_rating.numpy())

	return rating_pred_list, batch_rating_list


rating_pred_list, batch_rating_list = evaluate(mdl, dev_loader, use_cuda = args.use_cuda, phase = "Dev")
# these lists contains rating pred and batch rating for all batches, not just of some particular batch

# The above code is of pyTorchTESTPredict.py


# The below code is of Matrix.py
predicted_item_id = "B004MWZLYC"
expected_item_id = "B000IVUL64"
"""
predicted_item_id = "B004MWZLYC"
predicted_rating = 2.0

expected_item_id = "B000IVUL64"
expected_rating = 4.5
"""

item_df = pd.read_csv('out.csv',sep='\t', index_col=None)
# item_df.columns = ['MovieID', 'Name','Rating','Genre','Details']
# item_df = item_df.astype({'Rating':'float','Genre':'string','MovieID':'string'})
# item_df.dropna(how='any')
item_df = item_df.dropna()

item_df.head()


def matching_actuals_or_recommendations(predicted_item_id, predicted_rating, item_df):   # We're not using predicted_item_id, it's all use is commented.
	shortlisted_item_df = pd.DataFrame(columns=item_df.columns, index=None)
	# predicted_movie_features = item_df[item_df['MovieID']==predicted_item_id]['Genre'].tolist()
	# # print(predicted_movie_features)

	# for pred_mov_feature in predicted_movie_features[0].split(","):
	#   # print(pred_mov_feature)
	#   items_to_consider_df = item_df[item_df['Genre'].str.contains(pred_mov_feature, case=False)]
	#   # print(len(items_to_consider_df))
	#   items_to_consider_df = items_to_consider_df.drop_duplicates(subset=['Name','Rating'])
	#   shortlisted_item_df = shortlisted_item_df.append(items_to_consider_df,ignore_index=True)
	#   shortlisted_item_df = shortlisted_item_df.drop_duplicates(subset=['Name','Rating'])

	# shortlisted_item_df = shortlisted_item_df[shortlisted_item_df['Rating']>=predicted_rating]
	shortlisted_item_df = item_df[item_df['Rating'] >= predicted_rating].drop_duplicates(subset=['Rating'])

	return shortlisted_item_df[['MovieID', 'Rating']]


def precision_recall_f1_scores(predictions_df, actuals_df, k, rating_threshold):
	sorted_predictions_df = predictions_df.sort_values(by='Rating', ascending=False)
	sorted_actuals_df = actuals_df.sort_values(by='Rating', ascending=False)

	# Number of relevant items
	# n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
	num_rel_items = len(sorted_actuals_df[sorted_actuals_df['Rating'] >= rating_threshold])

	# print("Num of relevant items {}".format(num_rel_items))

	# Number of recommended items in top k
	# n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
	num_sorted_predictions = len(sorted_predictions_df)
	num_rec_k_items = len(sorted_predictions_df[:k]) if num_sorted_predictions >= k else num_sorted_predictions

	# print("Num of recommended k items {}".format(num_rec_k_items))

	# Number of relevant and recommended items in top k
	num_rel_and_rec_k_items = len(sorted_predictions_df[sorted_predictions_df['Rating'] >= rating_threshold][
								  :k]) if num_rec_k_items >= k else len(
		sorted_predictions_df[sorted_predictions_df['Rating'] >= rating_threshold])

	# print("Num of relevant and recommended k items {}".format(num_rel_and_rec_k_items))

	# Precision@K: Proportion of recommended items that are relevant
	# When n_rec_k is 0, Precision is undefined. We here set it to 0.
	precision = num_rel_and_rec_k_items / num_rec_k_items if num_rec_k_items != 0 else 0

	# Recall@K: Proportion of relevant items that are recommended
	# When n_rel is 0, Recall is undefined. We here set it to 0.

	recall = num_rel_and_rec_k_items / num_rel_items if num_rel_items != 0 else 0
	f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0 and recall == 0) else 0
	return precision, recall, f1, sorted_predictions_df[:k]

num_batches = 0
prec_all_batches = 0
rcl_all_batches = 0
f1sc_all_batches = 0
topk = -1  # just to initiate and store value of k in the below code to use outside of loop
for rating_pred, batch_rating in zip(rating_pred_list, batch_rating_list):
	size = 0  # size of each batch
	prec = 0
	rcl = 0
	f1Sc = 0
	num_batches += 1
	for predicted_rating, expected_rating in zip(rating_pred, batch_rating):
		matching_recommendations = matching_actuals_or_recommendations(predicted_item_id,predicted_rating,item_df)

		# for recommendation in matching_recommendations:
		#   print(recommendation)

		predictions_df = pd.DataFrame(matching_recommendations)

		# Convert rating column to float
		predictions_df['Rating'] = predictions_df['Rating'].astype(float)

		matching_actuals = matching_actuals_or_recommendations(expected_item_id,expected_rating,item_df)

		actuals_df = pd.DataFrame(matching_actuals)
		# Convert rating column to float
		actuals_df['Rating'] = actuals_df['Rating'].astype(float)
		k = 5
		topk = k
		precision, recall, f1, predictions_k = precision_recall_f1_scores(predictions_df, actuals_df, k, rating_threshold=expected_rating)

		# print("Precision@{0}: {1}, Recall@{0}: {2} and F1@{0} Score: {3}".format(k,precision,recall,f1))
		prec += precision
		rcl += recall
		f1Sc += f1
		size += 1
		# print("Expected Rating Threshold {0} and Predicted Ratings {1}".format(expected_rating, predictions_k))
		# The above code is of Matrix.py
	prec = prec / size
	rcl = rcl / size
	f1Sc = f1Sc / size
	prec_all_batches += prec
	rcl_all_batches += rcl
	f1sc_all_batches += f1Sc

precision = prec_all_batches / num_batches
recall = rcl_all_batches / num_batches
f1Score = f1sc_all_batches / num_batches
print("Precision@{0}: {1}, Recall@{0}: {2} and F1@{0} Score: {3}".format(topk,precision,recall,f1Score))