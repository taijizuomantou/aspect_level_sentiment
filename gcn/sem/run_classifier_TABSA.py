# coding=utf-8
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
--output_dir results/semeval2014/NLI_M \
--seed 42"""



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
import spacy
from spacy.gold import align
nlp = spacy.load("en_core_web_sm")
from My_model import BertForSequenceClassification
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_exist_id, aspect,noun_label,sent_label,head):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_exist_id = label_exist_id
        self.aspect = aspect
        self.noun_label = noun_label
        self.sent_label = sent_label
        self.head = head

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
    if typee == "train":
        #f = open("train_pos_tag.txt","r")
        f = open("new_train_tag.txt","r")
        fout = open("input_data/head_train.txt","r")
    else:
        f = open("new_test_tag.txt","r")
        fout = open("input_data/head_test.txt","r")
       #  f = open("test_pos_tag.txt","r")
   # f2 = open("test_text.txt","w")
   # f_new_noun = open("new_train_tag.txt","w")
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a,new_noun_label = tokenizer.tokenize(example.text_a)
#        text_a = " ".join(tokens_a)
#        doc = nlp(text_a)
#        spacy_text = []
#        for item in doc:
#            spacy_text.append(item.text)
#        alignment = align(tokens_a, spacy_text)
#        cost, a2b, b2a, a2b_multi, b2a_multi = alignment
#        temphead = len(tokens_a) * [-1]
#        k = 0;
#        for item in doc:
#            temphead[b2a[k]] = b2a[item.head.i]
#            k += 1
#        head = []
#        head.append(-1)
#        for item in temphead:
#            head.append(item + 1)
#        for item in head:
#            fout.write(str(item))
#            fout.write(" ")
#        fout.write("\n")
        temphead = fout.readline().split()
        head = []
        for item in temphead:head.append(int(item))
       # print(new_noun_label)
      #  print(tokens_a)
       # print(new_noun_label)
#
#        for i in range(len(new_noun_label) ):
#            item = new_noun_label[i]
#            #print(item)
#            #if item  == "NN" or item == "NNS" or item == "NNP" or item == "NNPS":
#            f_new_noun.write(item)
#            f_new_noun.write(" ")
#        f_new_noun.write("\n")
        str_noun_label = f.readline().split()
        #print(len(tokens_a))
        #print(len(str_noun_label))
        noun_label = []
        noun_label.append(0)
        for item in str_noun_label:
            if item  == "NN" or item == "NNS" or item == "NNP" or item == "NNPS":
                noun_label.append(1)
            else:
                noun_label.append(0)
        sent_label = []
        sent_label.append(0)
        for item in str_noun_label:
            if item  == "JJ" or item == "JJR"or item == "JJS" or item == "VB" or item == "VBD"or item == "VBG"or item == "VBN" or item == "VBP" or item == "VBZ" or item == "RB" or item == "RBR" or item == "RBS":
                sent_label.append(1)
            else:
                sent_label.append(0)
        
#        for i in range(len(tokens_a)):
#            #if noun_label[i] == 1:
#                f2.write(str(tokens_a[i]))
#                f2.write(" ")
#        f2.write("\n")
            
        tokens_b = None
        if example.text_b:
            tokens_b,_ = tokenizer.tokenize(example.text_b)
         #   print(tokens_b)
#            tokens_b = [example.text_b]
#            if example.text_b == "anecdotes":
#                tokens_b = ["other"]
#            if example.text_b == "ambience":
#                tokens_b = ["decor"]
          #  print(tokens_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
      #  print(tokens_b)
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
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        head.append(-1)
       # print(len(segment_ids))
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                #print(token)
                #print(tokenizer.convert_tokens_to_ids(tokens_b))
                tokens.append(token)
                segment_ids.append(1)
                head.append(-1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            head.append(-1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #pos_tag = nltk.pos_tag(tokens_a)
        #pos_tag = tokenizer.convert_text_to_tag(example.text_a)
      #  noun_label = []
       # noun_label.append(0)
       # noun_label.append(tokena_noun_label)
#        for item in pos_tag:
#            if item[1] == "NN" or item[1] == 'NNS' or item[1] == "NNP" or item[1] == "NNPS":
#                noun_label.append(1)
#            else:
#                noun_label.append(0)
      #  print(len(noun_label))
       # print(tokens_a)
        sent_label.append(0)
        noun_label.append(0)
        for item in range(len(tokens_b)):
            noun_label.append(1)
            sent_label.append(0)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        noun_label.append(0)
        sent_label.append(0)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
      #  print(len(noun_label))
      #  print()
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            noun_label.append(0)
            sent_label.append(0)
            head.append(-1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(noun_label) == max_seq_length
        assert len(sent_label) == max_seq_length
        assert len(head) == max_seq_length
       # print(tokens_b[0])
       # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        aspect = aspect_map[tokens_b[0]]
        label_id = label_map[example.label]
        label_exist_id = 1
        if label_id == 4: label_exist_id = 0
    #label_classify_id = []
#        for label in label_id:
#           if label == 4:
#                label_exist_id.append(0)
#           else:
#                label_exist_id.append(1)
        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        label_exist_id = label_exist_id,
                        aspect = aspect,
                        noun_label = noun_label,
                        sent_label = sent_label,
                        head = head
                        ))
    return features
def convert_examples_to_features_sentihood(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    aspect_map = {}
    aspect_list = ['price','an','food','am','service']
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    for (i, label) in enumerate(aspect_list):
        aspect_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
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
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
       # print(tokens_b[0])
       # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        my_dict = []
        #aspect = aspect_map[tokens_b[0]]
        my_dict[tokens_b[0]] += 1
        label_id = label_map[example.label]
        label_exist_id = 1
        if label_id == 0: label_exist_id = 0
        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        label_exist_id = label_exist_id,
                        #aspect = aspect
                        ))
    #print(my_dict)
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


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    
    ## Other parameters
    parser.add_argument("--eval_test",
                        default=False,
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
                        default=8,
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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)


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

    # training set
    train_examples = None
    num_train_steps = None
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size * args.num_train_epochs)

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer,"train")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)#乘完了iteration的次数

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all__detection_label_ids = torch.tensor([f.label_exist_id for f in train_features], dtype=torch.long)
    all_aspects= torch.tensor([f.aspect for f in train_features], dtype=torch.long)
    all_noun_label = torch.tensor([f.noun_label for f in train_features], dtype=torch.float)
    all_sent_label = torch.tensor([f.sent_label for f in train_features], dtype=torch.float)
    head = torch.tensor([f.head for f in train_features], dtype=torch.long)

    
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all__detection_label_ids,all_aspects,all_noun_label,all_sent_label,head)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # test set
    if args.eval_test:
        test_examples = processor.get_test_examples(args.data_dir)
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
        head = torch.tensor([f.head for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_detection_label_ids,all_aspects,all_noun_label,all_sent_label,head)
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
    #model = torch.load("model_data/new_sem/attention_add_model6")
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},#xiaobuxing
         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
         ]
   # temp_num_train_steps =  int(len(train_examples) / args.train_batch_size * 6.0)
    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    #optimizer = transformers.AdamW(optimizer_parameters)
    # train
    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=",output_log_file)
    with open(output_log_file, "w") as writer:
        if args.eval_test:
            writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
        else:
            writer.write("epoch\tglobal_step\tloss\n")
    
    global_step = 0
    epoch=0
   # model = torch.load("model_data/new_sem/attention_add_model6")
    #40
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        epoch+=1
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, exist_ids, all_aspects,all_noun_label,all_sent_label,head = batch
            loss, _ ,_= model(input_ids, segment_ids, input_mask, label_ids, exist_ids, all_aspects,all_noun_label,all_sent_label,head)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()    # We have accumulated enought gradients
                model.zero_grad()
                global_step += 1
        torch.save(model,"latestattention_add_model"+str(epoch))
           # break
          #  print(model.var)
         #eval_test
        if args.eval_test :
            model.eval()
            test_loss, test_accuracy,senti_test_accuracy = 0, 0,0
            nb_test_steps, nb_test_examples, senti_nb_test_examples = 0, 0, 0
            with open(os.path.join(args.output_dir, "test_ep_"+str(epoch)+".txt"),"w") as f_test:
                for input_ids, input_mask, segment_ids, label_ids, exist_ids,all_aspects,all_noun_label,all_sent_label,head in test_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    exist_ids = exist_ids.to(device)
                    all_aspects = all_aspects.to(device)
                    all_noun_label = all_noun_label.to(device)
                    all_sent_label = all_sent_label.to(device)
                    head = head.to(device)
                    with torch.no_grad():
                        tmp_test_loss, detect_logits,sent_logits = model(input_ids, segment_ids, input_mask, label_ids, exist_ids,all_aspects,all_noun_label,all_sent_label,head)

                    detect_logits = F.softmax(detect_logits, dim=-1)
                    detect_logits = detect_logits.detach().cpu().numpy()
                    exist_ids = exist_ids.to('cpu').numpy()
                    outputs = np.argmax(detect_logits, axis=1)
                    for output_i in range(len(outputs)):
                        f_test.write(str(outputs[output_i]))
                        for ou in detect_logits[output_i]:
                            f_test.write(" "+str(ou))
                        f_test.write("\n")
                    tmp_test_accuracy=np.sum(outputs == exist_ids)

                    test_loss += tmp_test_loss.mean().item()
                    test_accuracy += tmp_test_accuracy
                    
                    sent_logits = F.softmax(sent_logits, dim=-1)
                    sent_logits = sent_logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    senti_outputs = np.argmax(sent_logits, axis=1)
                    for output_i in range(len(senti_outputs)):
                        f_test.write(str(senti_outputs[output_i]))
                        for ou in sent_logits[output_i]:
                            f_test.write(" "+str(ou))
                        f_test.write("\n")
                    senti_tmp_test_accuracy=np.sum(senti_outputs == label_ids)

 
                    senti_test_accuracy += senti_tmp_test_accuracy

                    nb_test_examples += input_ids.size(0)
                    senti_nb_test_examples += np.sum(exist_ids)
                    nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples
            senti_accuracy = senti_test_accuracy / senti_nb_test_examples


        result = collections.OrderedDict()
        if args.eval_test:
            result = {'epoch': epoch,
                    'global_step': global_step,
                    'loss': tr_loss/nb_tr_steps,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'senti_test_accuracy':senti_accuracy,
                    }
        else:
            result = {'epoch': epoch,
                    'global_step': global_step,
                    'loss': tr_loss/nb_tr_steps}

        logger.info("***** Eval results *****")
        with open(output_log_file, "a+") as writer:
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")

if __name__ == "__main__":
    main()