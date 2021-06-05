# coding=utf-8




from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

import tokenization
import nltk
from model_load import BertForSequenceClassification
from optimization import BERTAdam
from processor import (Semeval_NLI_B_Processor, Semeval_NLI_M_Processor,
                       Semeval_QA_B_Processor, Semeval_QA_M_Processor,
                       Semeval_single_Processor, Sentihood_NLI_B_Processor,
                       Sentihood_NLI_M_Processor, Sentihood_QA_B_Processor,
                       Sentihood_QA_M_Processor, Sentihood_single_Processor)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_exist_id, aspect,noun_label,sent_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_exist_id = label_exist_id
        self.aspect = aspect
        self.noun_label = noun_label
        self.sent_label = sent_label


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,typee):
    """Loads a data file into a list of `InputBatch`s."""
    aspect_map = {}
  #  aspect_list = ['price','other','food','decor','service']
    aspect_list = ['price','an','food','am','service']
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    for (i, label) in enumerate(aspect_list):
        aspect_map[label] = i
  #  f = open("temp.txt","w")
    features = []
    example = examples

    tokens_a,new_noun_label = tokenizer.tokenize(example["text_a"])
    noun_label = []
    noun_label.append(0)
    for item in new_noun_label:
        if item  == "NN" or item == "NNS" or item == "NNP" or item == "NNPS":
            noun_label.append(1)
        else:
            noun_label.append(0)
    sent_label = []
    sent_label.append(0)
    for item in new_noun_label:
        if item  == "JJ" or item == "JJR"or item == "JJS" or item == "VB" or item == "VBD"or item == "VBG"or item == "VBN" or item == "VBP" or item == "VBZ" or item == "RB" or item == "RBR" or item == "RBS":
            sent_label.append(1)
        else:
            sent_label.append(0)

    tokens_b = None
    if example["text_b"]:
        tokens_b,_ = tokenizer.tokenize(example["text_b"])

    if tokens_b:

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
   # print(len(segment_ids))
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:

            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    sent_label.append(0)
    noun_label.append(0)
    for item in range(len(tokens_b)):
        noun_label.append(1)
        sent_label.append(1)

    noun_label.append(0)
    sent_label.append(0)
    input_mask = [1] * len(input_ids)


    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        noun_label.append(0)
        sent_label.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(noun_label) == max_seq_length
    assert len(sent_label) == max_seq_length

    aspect = aspect_map[tokens_b[0]]
    label_id = 0
    label_exist_id = 1
    if label_id == 4: label_exist_id = 0

    features.append(
            InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    label_exist_id = label_exist_id,
                    aspect = aspect,
                    noun_label = noun_label,
                    sent_label = sent_label
                    ))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
"""--task_name semeval_NLI_M \
--data_dir data/semeval2014/bert-pair/ \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 6.0 \
--output_dir results/semeval2014/NLI_M3 \
--seed 42"""
def main(doc,aspect):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default="semeval_NLI_M",
                        type=str,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default="data/semeval2014/bert-pair/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_file",
                        default="uncased_L-12_H-768_A-12/vocab.txt",
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_config_file",
                        default=" uncased_L-12_H-768_A-12/bert_config.json",
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--output_dir",
                        default="results/semeval2014/NLI_M3",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default="uncased_L-12_H-768_A-12/pytorch_model.bin",
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    
    ## Other parameters
    parser.add_argument("--eval_test",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the test set.")                    
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    args = parser.parse_args()


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    #bert_config = BertConfig.from_json_file(args.bert_config_file)

    #if args.max_seq_length > bert_config.max_position_embeddings:
     #   raise ValueError(
      #      "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
      #      args.max_seq_length, bert_config.max_position_embeddings))

   
    # prepare dataloaders
    processors = {
        "sentihood_single":Sentihood_single_Processor,
        "sentihood_NLI_M":Sentihood_NLI_M_Processor,
        "sentihood_QA_M":Sentihood_QA_M_Processor,
        "sentihood_NLI_B":Sentihood_NLI_B_Processor,
        "sentihood_QA_B":Sentihood_QA_B_Processor,
        "semeval_single":Semeval_single_Processor,
        "semeval_NLI_M":Semeval_NLI_M_Processor,
        "semeval_QA_M":Semeval_QA_M_Processor,
        "semeval_NLI_B":Semeval_NLI_B_Processor,
        "semeval_QA_B":Semeval_QA_B_Processor,
    }

    processor = processors[args.task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    


    # test set
    if args.eval_test:
        test_examples = {}
        test_examples["text_a"] = doc
        test_examples["text_b"] =aspect
        label_list =[]
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer,"test")

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_detection_label_ids = torch.tensor([f.label_exist_id for f in test_features], dtype=torch.long)
        all_aspects= torch.tensor([f.aspect for f in test_features], dtype=torch.long)
        all_noun_label = torch.tensor([f.noun_label for f in test_features], dtype=torch.float)
        all_sent_label = torch.tensor([f.sent_label for f in test_features], dtype=torch.float)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_detection_label_ids,all_aspects,all_noun_label,all_sent_label)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)


    # model and optimizer
    
    model = BertForSequenceClassification( len(label_list))
   # model = torch.load("model_data/attention_add_model")
    #if args.init_checkpoint is not None:
       # model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)



    

 #   model.load_state_dict(torch.load("my_50_model"))
    model.load_state_dict(torch.load("model_data/attention_4"))
   # model = torch.load("model_data/attention_add_model5")
    #40

    f = open("model_data/log13","w")

         #eval_test
    if args.eval_test :
        model.eval()

        for input_ids, input_mask, segment_ids, label_ids, exist_ids,all_aspects,all_noun_label,all_sent_label in test_dataloader:


            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            exist_ids = exist_ids.to(device)
            all_aspects = all_aspects.to(device)
            all_noun_label = all_noun_label.to(device)
            all_sent_label = all_sent_label.to(device)
            tmp_test_loss, detect_logits,sent_logits,attention_c,attention_d = model(input_ids, segment_ids, input_mask, label_ids, exist_ids,all_aspects,all_noun_label,all_sent_label)
           

            detect_logits = F.softmax(detect_logits, dim=-1)
            detect_logits = detect_logits.detach().cpu().numpy()
            exist_ids = exist_ids.to('cpu').numpy()
            outputs = np.argmax(detect_logits, axis=1)
            
            
            sent_logits = F.softmax(sent_logits, dim=-1)
            sent_logits = sent_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            senti_outputs = np.argmax(sent_logits, axis=1)
            print(outputs)
            print(senti_outputs)

    f.close()

if __name__ == "__main__":
    main()