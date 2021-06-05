# coding=utf-8

"""--task_name sentihood_NLI_M \
--data_dir data/sentihood/bert-pair/ \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 6.0 \
--output_dir results/sentihood/NLI_M \
--seed 42"""
from __future__ import absolute_import, division, print_function
from gcnModel import  Sentihood_model
import argparse
import collections
import logging
import os
import random
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
#import en_core_web_sm
import tokenization
import spacy
from spacy.gold import align
nlp = spacy.load("en_core_web_sm")
#from stanfordcorenlp import StanfordCoreNLP
#nlp = StanfordCoreNLP(r'/home/xue/stanford-corenlp-full-2018-10-05')
from Sentihood_model import BertForSequenceClassification
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_exist_id, aspect,my_pos,dis,my_pos_sent,noun_label,head):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_exist_id = label_exist_id
        self.aspect = aspect
        self.my_pos = my_pos
        self.dis = dis
        self.my_pos_sent = my_pos_sent
        self.noun_label = noun_label
        self.head= head
my_dict ={'NN': 0, 'JJ': 1, 'CD': 2, ':': 3, 'IN': 4, 'DT': 5, 'RB': 6, 'NNS': 7, 'CC': 8, 'VBP': 9, 'VBZ': 10, 'VB': 11, ',': 12, 'NNP': 13, 'PRP': 14, 'TO': 15, "''": 16, 'MD': 17, 'VBN': 18, 'VBD': 19, 'VBG': 20, 'PRP$': 21, ')': 22, '(': 23, 'JJR': 24, 'EX': 25, 'WDT': 26, 'JJS': 27, 'POS': 28, 'WRB': 29, 'RP': 30, 'RBR': 31, '``': 32, 'WP': 33, 'RBS': 34, 'FW': 35, 'PDT': 36, 'SYM': 37, '.': 38, 'UH': 39, '$': 40, 'NNPS': 41,'#':42}

def convert_examples_to_features_sentihood(examples, label_list, max_seq_length, tokenizer,typee):
   # f_train = open("senti_train.txt","w")
   # f_test = open("sent_test.txt","w")
    """Loads a data file into a list of `InputBatch`s."""
    aspect_map = {}
    aspect_list = ['location-1-general','location-1-price','location-1-safety','location-1-transitlocation','location-2-general','location-2-price','location-2-safety','location-2-transitlocation']
    #aspect_list = ['1general','1price','1safety','1transit','2general','2price','2safety','2transit']

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    for (i, label) in enumerate(aspect_list):
        aspect_map[label] = i

    features = []
   # my_dict = {}
    example = examples
        
    tokens_a,new_noun_label = tokenizer.tokenize(example["text_a"])
    if(typee == "test"):
        text_a = " ".join(tokens_a)
        doc = nlp(text_a)
        spacy_text = []
        for item in doc:
            spacy_text.append(item.text)
        alignment = align(tokens_a, spacy_text)
        cost, a2b, b2a, a2b_multi, b2a_multi = alignment
        temphead = len(tokens_a) * [-1]
        k = 0;
        for item in doc:
            temphead[b2a[k]] = b2a[item.head.i]
            k += 1
        head = []
        head.append(-1)
        for item in temphead:
            head.append(item + 1)
#            for item in head:
#                fout.write(str(item))
#                fout.write(" ")
#            fout.write("\n")
#        else:


    tokens_b = None
    if example["text_b"]:
       # if(example.text_b == "location - 1 - general"):example.text_b = "location - 1 - overall"
       # if(example.text_b == "location - 2 - general"):example.text_b = "location - 1\ - overall"
        tokens_b,_ = tokenizer.tokenize(example["text_b"])


        if len(tokens_a) > max_seq_length -9:
            tokens_a = tokens_a[0:(max_seq_length - 9)]
            head = head[0:(max_seq_length - 8)]
            new_noun_label = new_noun_label[0:(max_seq_length - 9)]#if generate cixing f xuyao 
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    head.append(-1)
    segment_ids.append(0)
  #  print(tokens_a)
#        iterate = 0
#        new_tokens_b = []
#        if tokens_b:
#            for token in tokens_b:
#                if(iterate == 0 or iterate == 2 or iterate == 4):new_tokens_b.append(token)
#                iterate += 1
#        tokens_b = new_tokens_b
  #  print(tokens_b)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
            head.append(-1)

            
        tokens.append("[SEP]")
        segment_ids.append(1)
        head.append(-1)
    #nlp = en_core_web_sm.load()
    #docs = nlp(str(tokens))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    aspect_index =  tokenizer.convert_tokens_to_ids(tokens_b)
    #print(tokens_b)
    #print(aspect_index)

#        pos_tag = new_noun_label
#       # print(pos_tag)
#        if typee == "train":
#            f_cixing = f_train
#            for i in range(len(pos_tag) ):
#                item = pos_tag[i]
#                #print(item)
#                #if item  == "NN" or item == "NNS" or item == "NNP" or item == "NNPS":
#                f_cixing.write(item)
#                f_cixing.write(" ")
#            f_cixing.write("\n")
#        elif typee == "test":
#            f_cixing = f_test
#            for i in range(len(pos_tag) ):
#                item = pos_tag[i]
#                #print(item)
#                #if item  == "NN" or item == "NNS" or item == "NNP" or item == "NNPS":
#                f_cixing.write(item)
#                f_cixing.write(" ")
#            f_cixing.write("\n")
#        if typee == "train":
#        #f = open("train_pos_tag.txt","r")
#            f = open("senti_train.txt","r")
#        else:
#            f = open("senti_test.txt","r")
#        pos_tag = f.readline().split()
    pos_tag =new_noun_label
#        my_pos = []
    noun_label = []
    noun_label.append(0)
    sent_label = []
    sent_label.append(0)
#        for item in pos_tag:
#            temp = my_dict[item]+1
#            if temp > 1:temp = 0
#            my_pos.append(temp)
#      #  my_pos_sent = []

# pay attention that only cd may be better
    for item in pos_tag:
        if item == "NN" or item == 'NNS' or item== "NNP" or item == "NNPS" or item == 'CD' or item ==':':
            noun_label.append(1)
        else:
            noun_label.append(0)
        if item  == "JJ" or item == "JJR"or item == "JJS" or item == "VB" or item == "VBD"or item == "VBG"or item == "VBN" or item == "VBP" or item == "VBZ" or item == "RB" or item == "RBR" or item == "RBS":
            sent_label.append(1)
        else:
            sent_label.append(0)
      #  if item[1] == ":":
           # print(item[0])
      #  print(item)
    # T#he mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    dis = []
    number = tokens_b[2]
    begin_id = -1
    for i in range(len(tokens)):
        if tokens_a[i] == "location"and tokens_a[i + 1] == '-' and tokens_a[i + 2] == number:
            begin_id = i
            break
    for i in range(len(tokens)):
        if i == begin_id or i == begin_id + 1 or i == begin_id + 2 :dis.append(0)
        elif i < begin_id: dis.append((begin_id - i))
        else: dis.append((i - begin_id - 2))
   # print(dis)
    # Zero-pad up to the sequence length.
    sent_label.append(0)
    noun_label.append(0)
    for item in range(len(tokens_b)):
        noun_label.append(1)
        sent_label.append(0)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    noun_label.append(0)
    sent_label.append(0)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
       # my_pos.append(0)
        #my_pos_sent.append(0)
        dis.append(0)
        head.append(-1)
        noun_label.append(0)
        sent_label.append(0)
  #  while len(dis) < max_seq_length:
        
 #   print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(dis) == max_seq_length
  #  assert len(my_pos_sent) == max_seq_length
    
    assert len(noun_label) == max_seq_length
    
    assert len(sent_label) == max_seq_length
   # print(tokens_b[0])
   # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    assert len(head) == max_seq_length
    #aspect = aspect_map[tokens_b[0]]
   # tempb =tokens_b[4]
  #  print(tokens_b)
    
    tempb = ""
    for item in tokens_b:
        tempb += item

  #  print(tempb)
    
    label_id = 0
    aspect = aspect_map[tempb]
    label_exist_id = 1
  #  print(my_pos)
   # print(input_ids)
    if label_id == 2: label_exist_id = 0
    features.append(
            InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    label_exist_id = label_exist_id,
                    aspect = aspect,
                    my_pos = _,
                    dis = dis,
                    my_pos_sent = sent_label,#pay  attention change in this part
                    noun_label = noun_label,
                    head= head
                    ))
        
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


def main(doc,aspect):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default="sentihood_NLI_M",
                        type=str,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default="data/sentihood/bert-pair/",
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
        test_examples["text_b"] = aspect
        label_list =[]
        test_features = convert_examples_to_features_sentihood(
            test_examples, label_list, args.max_seq_length, tokenizer,"test")

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_detection_label_ids = torch.tensor([f.label_exist_id for f in test_features], dtype=torch.long)
        all_aspects= torch.tensor([f.aspect for f in test_features], dtype=torch.long)
        all_pos = torch.tensor([f.aspect for f in test_features], dtype=torch.long)
        all_dis = torch.tensor([f.dis for f in test_features], dtype=torch.long)
        all_pos_sent = torch.tensor([f.my_pos_sent for f in test_features], dtype=torch.float)
        all_noun_label = torch.tensor([f.noun_label for f in test_features], dtype=torch.float)
        head = torch.tensor([f.head for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_detection_label_ids,all_aspects,all_pos,all_dis,all_pos_sent,all_noun_label,head)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)


    # model and optimizer
    
    model = Sentihood_model.BertForSequenceClassification( len(label_list))
   # model = torch.load("model_data/attention_add_model")
    #if args.init_checkpoint is not None:
       # model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    model.load_state_dict(torch.load("model_data/gcn/senti"))
#    paras = dict(model.named_parameters())
#    bert_no_decay = []
#    bert_decay = []
#    no_bert_no_decay = []
#    no_bert_decay = []
#    for k, v in paras.items():
#         if 'bias' in k:
#            if 'bert' in k:
#                 bert_no_decay.append(v)
#            else:
#                 no_bert_no_decay.append(v)
#         else:
#            if 'bert' in k:
#                 bert_decay.append(v)
#            else:
#                 no_bert_decay.append(v)
#    lr = 3e-5
#    big_lr = 0.2
#
#    optimizer_parameters = [
#         {'params': bert_no_decay, 'lr':lr, 'weight_decay_rate': 0.0},#xiaobuxing
#         {'params': bert_decay, 'lr':lr, 'weight_decay_rate': 0.01},#xiaobuxing
#         {'params': no_bert_no_decay, 'lr':big_lr, 'weight_decay_rate': 0.0},#xiaobuxing
#         {'params': no_bert_decay, 'lr':big_lr, 'weight_decay_rate': 0.01},#xiaobuxing
#         ]
		

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
    resexist = ""
    ressent = ""
   # model = torch.load("model_data/attention_add_model6")
    #40
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        
         #eval_test
        if args.eval_test :
            model.eval()
            test_loss, test_accuracy,senti_test_accuracy = 0, 0,0
            nb_test_steps, nb_test_examples, senti_nb_test_examples = 0, 0, 0
            with open(os.path.join(args.output_dir, "test_ep_"+str(epoch)+".txt"),"w") as f_test:
                for input_ids, input_mask, segment_ids, label_ids, exist_ids,all_aspects,pos,dis,all_pos_sent,all_noun_label,head in test_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    exist_ids = exist_ids.to(device)
                    all_aspects = all_aspects.to(device)
                    dis = dis.to(device)
                    pos = pos.to(device)
                    all_noun_label = all_noun_label.to(device)
                    all_pos_sent = all_pos_sent.to(device)
                    head = head.to(device)
                    with torch.no_grad():
                        tmp_test_loss, detect_logits,sent_logits = model(input_ids, segment_ids, input_mask, label_ids, exist_ids, all_aspects,all_noun_label,all_pos_sent,head)

                    detect_logits = F.softmax(detect_logits, dim=-1)
                    detect_logits = detect_logits.detach().cpu().numpy()
                    exist_ids = exist_ids.to('cpu').numpy()
                    outputs = np.argmax(detect_logits, axis=1)

                    sent_logits = F.softmax(sent_logits, dim=-1)
                    sent_logits = sent_logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    senti_outputs = np.argmax(sent_logits, axis=1)
                    resexist = outputs[0]
                    ressent = senti_outputs[0]
    return resexist,ressent

if __name__ == "__main__":
    print(main("i think location - 1 is bad","location-1-general"))