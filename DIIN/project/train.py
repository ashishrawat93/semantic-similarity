
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import importlib
import random

from helper.data_processing import *
from my_model import evaluate_classifier
from tqdm import tqdm
import gzip
import pickle
import shutil

LABEL_MAP = {'entailment':1 , 'nonentailment':0}

hyperparameters = {
        "model_type": "my_model",
        "model_name": "demo_testing_SNLI",
        "training_mnli": "{}/multinli_0.9/multinli_0.9_train.jsonl",
        "dev_matched": "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl",
        "dev_mismatched": "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl",
        "test_matched": "{}/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl",
        "test_mismatched": "{}/multinli_0.9/multinli_0.9_test_mismatched_unlabeled.jsonl",
        "training_snli": "../data/snli_1.0/snli_1.0_train.jsonl",
        "dev_snli": "../data/snli_1.0/snli_1.0_dev.jsonl",
        "test_snli": "../data/snli_1.0/snli_1.0_test.jsonl",
        "embedding_data_path": "../data/glove.840B.300d.txt",
        "log_path": "",
        "ckpt_path":  "",
        "embeddings_to_load": None,
        "word_embedding_dim": 300,
        "hidden_embedding_dim": 300,
        "seq_length": 48,
        "keep_rate": 1.0, 
        "batch_size": 70,
        "learning_rate": 0.5,
        "emb_train": "store_false",
        "alpha": 0.15,
        "genre": ['travel', 'fiction', 'slate', 'telephone', 'government'],
        "tbpath": "",
        "test": "store_true",
        "debug_model": "store_true",
        "data_path": "../data",
        "char_in_word_size": 16,
        "evalstep": 1000,
        "data" : "../data",
        "char_emb_size": 8,
        "highway_num_layers":2,
        "wd" :0.0,
        "first_scale_down_layer_relu": "store_true",
        "out_channel_dims": "100",
        "filter_heights": "5",
        "char_out_size": 100,
        "self_att_logit_func": "tri_linear",
        "self_att_fuse_gate_residual_conn": "store_false",
        "self_att_fuse_gate_relu_z":"store_true",
        "two_gate_fuse_gate": "store_false"

    }

modname = hyperparameters["model_name"]



model = hyperparameters["model_type"]

module = importlib.import_module(".".join([model])) 
MyModel = getattr(module, 'DIIN')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
print("parameters\n %s" % hyperparameters)

filename = os.path.join(hyperparameters["ckpt_path"], modname) + '_checkpoint.pth.tar'
best_model_path = os.path.join(hyperparameters["ckpt_path"], modname) + '_model_best.pth.tar'

######################### LOAD DATA #############################


if hyperparameters["debug_model"]:
    test_matched = load_nli_data(hyperparameters["dev_snli"], shuffle = False)[:499]
    training_snli, dev_snli, test_snli = test_matched, test_matched,test_matched
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([test_matched])
    shared_content = load_mnli_shared_content()
else:

    print("Loading data SNLI")
    training_snli = load_nli_data(hyperparameters["training_snli"], snli=True)
    dev_snli = load_nli_data(hyperparameters["dev_snli"], snli=True)
    test_snli = load_nli_data(hyperparameters["test_snli"], snli=True)
   
    
    print("Loading embeddings")
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([training_snli, dev_snli, test_snli])
    shared_content = load_mnli_shared_content()

hyperparameters["char_vocab_size"] = len(char_indices.keys())

embedding_dir = os.path.join(hyperparameters["data"], "embeddings")
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)


embedding_path = os.path.join(embedding_dir, "mnli_emb_snli_embedding.pkl.gz")

print("embedding path exist")
print(os.path.exists(embedding_path))
if os.path.exists(embedding_path):
   
    f = gzip.open(embedding_path, 'rb')
  
    loaded_embeddings = pickle.load(f)

    f.close()
else:
 
    loaded_embeddings = loadEmbedding_rand(hyperparameters["embedding_data_path"], word_indices)
    f = gzip.open(embedding_path, 'wb')
    pickle.dump(loaded_embeddings, f)
    f.close()

def get_minibatch(dataset, start_index, end_index, training=False):
    indices = range(start_index, end_index)

    labels = [dataset[i]['label'] for i in indices]
    pairIDs = np.array([dataset[i]['pairID'] for i in indices])


    premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] * len(indices)

    premise_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_index_sequence'][:] for i in indices], premise_pad_crop_pair, 1)
    hypothesis_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_index_sequence'][:] for i in indices], hypothesis_pad_crop_pair, 1)
    premise_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_char_index'][:] for i in indices], premise_pad_crop_pair, 2, column_size=hyperparameters["char_in_word_size"])
    hypothesis_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_char_index'][:] for i in indices], hypothesis_pad_crop_pair, 2, column_size=hyperparameters["char_in_word_size"])

    premise_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence1_parse'][:] for i in indices], premise_pad_crop_pair)
    hypothesis_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence2_parse'][:] for i in indices], hypothesis_pad_crop_pair)

    premise_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence1_token_exact_match_with_s2"][:] for i in range(len(indices))], premise_pad_crop_pair, 1)
    hypothesis_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence2_token_exact_match_with_s1"][:] for i in range(len(indices))], hypothesis_pad_crop_pair, 1)

    premise_exact_match = np.expand_dims(premise_exact_match, 2)
    hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)

    labels = torch.LongTensor(labels)

    minibatch_premise_vectors = torch.stack([torch.from_numpy(v) for v in premise_vectors]).squeeze().type('torch.LongTensor')
    minibatch_hypothesis_vectors = torch.stack([torch.from_numpy(v) for v in hypothesis_vectors]).squeeze().type('torch.LongTensor')

    minibatch_pre_pos = torch.stack([torch.from_numpy(v) for v in premise_pos_vectors]).squeeze().type('torch.FloatTensor')
    minibatch_hyp_pos = torch.stack([torch.from_numpy(v) for v in hypothesis_pos_vectors]).squeeze().type('torch.FloatTensor')

    premise_char_vectors = torch.stack([torch.from_numpy(v) for v in premise_char_vectors]).squeeze().type('torch.LongTensor')
    hypothesis_char_vectors = torch.stack([torch.from_numpy(v) for v in hypothesis_char_vectors]).squeeze().type('torch.LongTensor')
    
    premise_exact_match = torch.stack([torch.from_numpy(v) for v in premise_exact_match]).squeeze().type('torch.FloatTensor')
    hypothesis_exact_match = torch.stack([torch.from_numpy(v) for v in hypothesis_exact_match]).squeeze().type('torch.FloatTensor')

    return minibatch_premise_vectors, minibatch_hypothesis_vectors, labels, \
        minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
        premise_exact_match, hypothesis_exact_match


def train(model, loss_, optim, batch_size,hyperparameters , train_snli, dev_snli):
    display_epoch_freq = 1
    display_step = 1
    eval_step = 1000
    save_step = 1000
    embedding_dim = hyperparameters["word_embedding_dim"]
    dim = hyperparameters["hidden_embedding_dim"]
    emb_train = hyperparameters["emb_train"]
    keep_rate = hyperparameters["keep_rate"]
    sequence_length = hyperparameters["seq_length"] 
    alpha = hyperparameters["alpha"]
    # config = config

    print("Building model from %s.py" %(model))
    model.train()

    print("Initializing variables")

    step = 1
    epoch = 0
    best_dev_mat = 0.
    best_mtrain_acc = 0.
    last_train_acc = [.001, .001, .001, .001, .001]
    best_step = 0
    train_dev_set = False
    dont_print_unnecessary_info = False
    collect_failed_sample = False
    # Restore most recent checkpoint if it exists. 
    # Also restore values for best dev-set accuracy and best training-set accuracy

    if os.path.isfile(filename):
        if os.path.isfile(best_model_path):
            #self.saver.restore(self.sess, (ckpt_file + "_best"))
            print("=> loading checkpoint '{}'".format(best_model_path))
            checkpoint = torch.load(best_model_path)
            epoch = checkpoint['epoch']
            best_dev_snli = checkpoint['best_prec1']
            print("Saved best SNLI-dev acc: %f" %best_dev_snli)
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("Finish load model")
            completed = False
            best_dev_snli, dev_cost_snli, confmx = evaluate_classifier(classify, dev_snli, batch_size, completed, model, loss_)
            print("Confusion Matrix on dev-snli\n{}".format(confmx))
            print("Restored best SNLI-dev acc: %f" %(best_dev_snli))
            best_dev_mat = best_dev_snli
        else:
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            best_dev_snli = checkpoint['best_prec1']
            print("Saved checkpoint SNLI-dev acc: %f" %best_dev_snli)
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("Finish load model")
            completed = False
            best_dev_snli, dev_cost_snli, confmx = evaluate_classifier(classify, dev_snli, batch_size, completed, model, loss_)
            print("Confusion Matrix on dev-snli\n{}".format(confmx))
            print("Restored best SNLI-dev acc: %f" %(best_dev_snli))
            best_dev_mat = best_dev_snli

    ### Training cycle
    print("Training...")

    while True:
        training_data = train_snli

        random.shuffle(training_data)
        avg_cost = 0.
        total_batch = int(len(training_data) / batch_size)
        
        # Boolean stating that training has not been completed, 
        completed = False 

        # Loop over all batches in epoch
        for i in range(total_batch):

            # Assemble a minibatch of the next B examples
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
            minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match  = get_minibatch(
            training_data, batch_size * i, batch_size * (i + 1), True)
                    

            minibatch_premise_vectors = Variable(minibatch_premise_vectors)
            minibatch_hypothesis_vectors = Variable(minibatch_hypothesis_vectors)

            minibatch_pre_pos = Variable(minibatch_pre_pos)
            minibatch_hyp_pos = Variable(minibatch_hyp_pos)

            premise_char_vectors = Variable(premise_char_vectors)
            hypothesis_char_vectors = Variable(hypothesis_char_vectors)
            premise_exact_match = Variable(premise_exact_match)
            hypothesis_exact_match = Variable(hypothesis_exact_match)

            minibatch_labels = Variable(minibatch_labels)

            model.zero_grad()
            # Run the optimizer to take a gradient step, and also fetch the value of the 
            # cost function for logging

            model.dropout_rate_decay(step)
            output = model(minibatch_premise_vectors, minibatch_hypothesis_vectors, \
                minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match)
            print("Finish forward")
            print("Finish forward{}".format(step))
            lossy = loss_(output, minibatch_labels)

            
            diff_loss = F.mse_loss(model.self_attention_linear_p.weight.data, model.self_attention_linear_h.weight.data) * torch.numel(model.self_attention_linear_p.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p1.weight.data, model.fuse_gate_linear_h1.weight.data) * torch.numel(model.fuse_gate_linear_p1.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p2.weight.data, model.fuse_gate_linear_h2.weight.data) * torch.numel(model.fuse_gate_linear_p2.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p3.weight.data, model.fuse_gate_linear_h3.weight.data) * torch.numel(model.fuse_gate_linear_p3.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p4.weight.data, model.fuse_gate_linear_h4.weight.data) * torch.numel(model.fuse_gate_linear_p4.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p5.weight.data, model.fuse_gate_linear_h5.weight.data) * torch.numel(model.fuse_gate_linear_p5.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p6.weight.data, model.fuse_gate_linear_h6.weight.data) * torch.numel(model.fuse_gate_linear_p6.weight.data) / 2.0

            diff_loss *= 1e-3

            lossy += diff_loss

            print("loss{}".format(lossy.item()))
            lossy.backward()
            print("Finish backward{}".format(step))
            print("Finish backward{}".format(step))
            torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), 1)
            optim.step()

            if step % display_step == 0:
                print("Step: {} completed".format(step))
                print("Step: {} completed".format(step))
            if step % eval_step == 1:
                print("start eval dev_snli:") 
                print("start eval dev_snli:")
                dev_acc_snli, dev_cost_snli, confmx = evaluate_classifier(classify, dev_snli, batch_size, completed, model, loss_)
                print("Confusion Matrix on dev_snli\n{}".format(confmx))

                strain_acc, strain_cost, confmx= evaluate_classifier(classify, train_snli[0:5000], batch_size, completed, model, loss_)
                print("Confusion Matrix on train_snli\n{}".format(confmx))
                print("Step: %i\t Dev-SNLI acc: %f\t SNLI train acc: %f" %(step, dev_acc_snli, strain_acc))
                print("Step: %i\t Dev-SNLI cost: %f\t SNLI train cost: %f" %(step, dev_cost_snli, strain_cost))

                print("Finish eval")

            if step % save_step == 1:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': dev_acc_snli,
                    'optimizer' : optim.state_dict(),
                    }, filename)
                
                dev_acc_mat = dev_acc_snli
                best_test = 100 * (1 - best_dev_mat / dev_acc_mat)
                if best_test > 0.04:
                    shutil.copyfile(filename, best_model_path)
                    best_dev_mat = dev_acc_mat
                    if alpha != 0.:
                        best_strain_acc = strain_acc
                    best_step = step
                    print("Checkpointing with new best matched-dev accuracy: %f" %(best_dev_mat))

            if best_dev_mat > 0.872 :
                eval_step = 500
                save_step = 500
            
            if best_dev_mat > 0.878 :
                eval_step = 100
                save_step = 100
                dont_print_unnecessary_info = True 

            step += 1

            # Compute average loss
            avg_cost += lossy.item() / (total_batch * batch_size)
                            
        # Display some statistics about the epoch
        if epoch % display_epoch_freq == 0:
            print("Epoch: %i\t Avg. Cost: %f" %(epoch+1, avg_cost))
        print("Epoch: %i\t Avg. Cost: %f" %(epoch+1, avg_cost))
        epoch += 1 

        last_train_acc[(epoch % 5) - 1] = strain_acc

        # Early stopping
        early_stopping_step = 35000
        progress = 1000 * (sum(last_train_acc)/(5 * min(last_train_acc)) - 1) 

        
        if (progress < 0.1) or (step > best_step + early_stopping_step):
            print("Best dev accuracy: %s" %(best_dev_mat))
            print("SNLI Train accuracy: %s" %(best_strain_acc))
            
            train_dev_set = True
            completed = True
             


def classify(examples, completed, batch_size, model, loss_):
    model.eval()
    # This classifies a list of examples
    if (test == True) or (completed == True):
        checkpoint = torch.load(best_model_path)
        epoch = checkpoint['epoch']
        best_dev_snli = checkpoint['best_prec1']
        print("Saved best SNLI-dev acc: %f" %best_dev_snli)
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        print("Model restored from file: %s" % best_model_path)

    total_batch = int(len(examples) / batch_size)
    pred_size = 3 
    logits = np.empty(pred_size)
    genres = []
    costs = 0
    correct = 0
    for i in range(total_batch):
        minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
        minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
        premise_exact_match, hypothesis_exact_match  = get_minibatch(
            examples, batch_size * i, batch_size * (i + 1))

        minibatch_premise_vectors = Variable(minibatch_premise_vectors)
        minibatch_hypothesis_vectors = Variable(minibatch_hypothesis_vectors)

        minibatch_pre_pos = Variable(minibatch_pre_pos)
        minibatch_hyp_pos = Variable(minibatch_hyp_pos)

        premise_char_vectors = Variable(premise_char_vectors)
        hypothesis_char_vectors = Variable(hypothesis_char_vectors)
        premise_exact_match = Variable(premise_exact_match)
        hypothesis_exact_match = Variable(hypothesis_exact_match)

        minibatch_labels = Variable(minibatch_labels)

        logit = model(minibatch_premise_vectors, minibatch_hypothesis_vectors, \
            minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match)

        cost = loss_(logit, minibatch_labels).item()
        costs += cost

        logits = np.vstack([logits, logit.data.numpy()])
    return genres, np.argmax(logits[1:], axis=1), costs

def generate_predictions_with_id(path, examples, completed, batch_size, model, loss_):
    if (test == True) or (completed == True):
        checkpoint = torch.load(best_model_path)
        epoch = checkpoint['epoch']
        best_dev_snli = checkpoint['best_prec1']
        print("Saved best SNLI-dev acc: %f" %best_dev_snli)
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        print("Model restored from file: %s" % best_model_path)

    total_batch = int(len(examples) / batch_size)
    pred_size = 3
    logits = np.empty(pred_size)
    costs = 0
    IDs = np.empty(1)
    for i in range(total_batch):
        minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
        minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
        premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
        hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
        hypothesis_NER_feature  = get_minibatch(
            examples, batch_size * i, batch_size * (i + 1))

        minibatch_premise_vectors = Variable(minibatch_premise_vectors)
        minibatch_hypothesis_vectors = Variable(minibatch_hypothesis_vectors)

        minibatch_pre_pos = Variable(minibatch_pre_pos)
        minibatch_hyp_pos = Variable(minibatch_hyp_pos)

        premise_char_vectors = Variable(premise_char_vectors)
        hypothesis_char_vectors = Variable(hypothesis_char_vectors)
        premise_exact_match = Variable(premise_exact_match)
        hypothesis_exact_match = Variable(hypothesis_exact_match)

        minibatch_labels = Variable(minibatch_labels)

        logit = model(minibatch_premise_vectors, minibatch_hypothesis_vectors, \
            minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match)
        IDs = np.concatenate([IDs, pairIDs])
        logits = np.vstack([logits, logit])
    IDs = IDs[1:]
    logits = np.argmax(logits[1:], axis=1)
    save_submission(path, IDs, logits)

batch_size = hyperparameters["batch_size"]
completed = False



model = MyModel(hyperparameters, hyperparameters["seq_length"], emb_dim=hyperparameters["word_embedding_dim"],  hidden_dim=hyperparameters["hidden_embedding_dim"], embeddings=loaded_embeddings, emb_train=hyperparameters["emb_train"])


optim = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr = hyperparameters["learning_rate"])
loss = nn.CrossEntropyLoss() 

test = params.train_or_test()
print('test'.format(test))



train(model, loss, optim, batch_size, hyperparameters, training_snli, dev_snli)
completed = True
test_acc_snli, test_cost_snli, confmx = evaluate_classifier(classify, test_snli, hyperparameters["batch_size"], completed, model, loss)

print("Confusion Matrix on test_snli\n{}".format(confmx))
print("Acc on SNLI test-set: %f" %(test_acc_snli))

    
