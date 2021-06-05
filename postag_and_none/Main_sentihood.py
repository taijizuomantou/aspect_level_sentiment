# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_exist_id, aspect,my_pos,dis,my_pos_sent,noun_label):
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
my_dict ={'NN': 0, 'JJ': 1, 'CD': 2, ':': 3, 'IN': 4, 'DT': 5, 'RB': 6, 'NNS': 7, 'CC': 8, 'VBP': 9, 'VBZ': 10, 'VB': 11, ',': 12, 'NNP': 13, 'PRP': 14, 'TO': 15, "''": 16, 'MD': 17, 'VBN': 18, 'VBD': 19, 'VBG': 20, 'PRP$': 21, ')': 22, '(': 23, 'JJR': 24, 'EX': 25, 'WDT': 26, 'JJS': 27, 'POS': 28, 'WRB': 29, 'RP': 30, 'RBR': 31, '``': 32, 'WP': 33, 'RBS': 34, 'FW': 35, 'PDT': 36, 'SYM': 37, '.': 38, 'UH': 39, '$': 40, 'NNPS': 41,'#':42}

def convert_examples_to_features_sentihood(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    aspect_map = {}
    aspect_list = ['location-1-general','location-1-price','location-1-safety','location-1-transitlocation','location-2-general','location-2-price','location-2-safety','location-2-transitlocation']
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    for (i, label) in enumerate(aspect_list):
        aspect_map[label] = i

    features = []
   # my_dict = {}
    for (ex_index, example) in enumerate(tqdm(examples)):
        
        tokens_a,_ = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b,_ = tokenizer.tokenize(example.text_b)

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
        #nlp = en_core_web_sm.load()
        #docs = nlp(str(tokens))
      #  print(tokens)
        #print("!!!!!!!!!!!!!!!!!!")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pos_tag = nltk.pos_tag(tokens)
        my_pos = []
        noun_label = []
        for item in pos_tag:
            temp = my_dict[item[1]]+1
            if temp > 1:temp = 0
            my_pos.append(temp)
        my_pos_sent = []
        for item in pos_tag:
            if item[1] == 'JJ':
                temp = 1
            else:temp = 0
            my_pos_sent.append(temp)
        for item in pos_tag:
            if item[1] == "NN" or item[1] == 'NNS' or item[1] == "NNP" or item[1] == "NNPS":
                noun_label.append(1)
            else:
                noun_label.append(0)
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
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            my_pos.append(0)
            my_pos_sent.append(0)
            dis.append(0)
            noun_label.append(0)
      #  while len(dis) < max_seq_length:
            

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(dis) == max_seq_length
        assert len(my_pos_sent) == max_seq_length
        assert len(noun_label) == max_seq_length
       # print(tokens_b[0])
       # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        #aspect = aspect_map[tokens_b[0]]
       # tempb =tokens_b[4]
      #  print(tokens_b)
        
        tempb = ""
        for item in tokens_b:
            tempb += item
      #  print(tempb)
        
        label_id = label_map[example.label]
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
                        my_pos = my_pos,
                        dis = dis,
                        my_pos_sent = my_pos_sent,
                        noun_label = noun_label
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
                        default=4,
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

#--task_name sentihood_NLI_M \
#--data_dir data/sentihood/bert-pair/ \
#--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
#--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
#--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
#--eval_test \
#--do_lower_case \
#--max_seq_length 128 \
#--train_batch_size 2 \
#--learning_rate 2e-5 \
#--num_train_epochs 6.0 \
#--output_dir senti_results/sentihood/NLI_M \
#--seed 42
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

    train_features = convert_examples_to_features_sentihood(
        train_examples, label_list, args.max_seq_length, tokenizer)
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
    all_pos = torch.tensor([f.my_pos for f in train_features], dtype=torch.long)
    all_dis = torch.tensor([f.dis for f in train_features], dtype=torch.long)
    all_pos_sent = torch.tensor([f.my_pos_sent for f in train_features], dtype=torch.long)
    all_noun_label = torch.tensor([f.noun_label for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all__detection_label_ids,all_aspects,all_pos,all_dis,all_pos_sent,all_noun_label)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # test set
    if args.eval_test:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features_sentihood(
            test_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_detection_label_ids = torch.tensor([f.label_exist_id for f in test_features], dtype=torch.long)
        all_aspects= torch.tensor([f.aspect for f in test_features], dtype=torch.long)
        all_pos = torch.tensor([f.my_pos for f in test_features], dtype=torch.long)
        all_dis = torch.tensor([f.dis for f in test_features], dtype=torch.long)
        all_pos_sent = torch.tensor([f.my_pos_sent for f in test_features], dtype=torch.long)
        all_noun_label = torch.tensor([f.noun_label for f in test_features], dtype=torch.float)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_detection_label_ids,all_aspects,all_pos,all_dis,all_pos_sent,all_noun_label)
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

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},#xiaobuxing
         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
         ]
		
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
   # model = torch.load("model_data/attention_add_model6")
    #40
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        epoch+=1
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
           # print()
            input_ids, input_mask, segment_ids, label_ids, exist_ids, all_aspects,pos,dis,all_pos_sent,all_noun_label = batch
            loss, _ ,_= model(input_ids, segment_ids, input_mask, label_ids, exist_ids, all_aspects,pos,dis,all_pos_sent,all_noun_label)
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
           # break
        torch.save(model,"model_data/sentihood_attention_add_model"+str(epoch))
         #eval_test
        if args.eval_test :
            model.eval()
            test_loss, test_accuracy,senti_test_accuracy = 0, 0,0
            nb_test_steps, nb_test_examples, senti_nb_test_examples = 0, 0, 0
            with open(os.path.join(args.output_dir, "test_ep_"+str(epoch)+".txt"),"w") as f_test:
                for input_ids, input_mask, segment_ids, label_ids, exist_ids,all_aspects,pos,dis,all_pos_sent,all_noun_label in test_dataloader:
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
                    with torch.no_grad():
                        tmp_test_loss, detect_logits,sent_logits = model(input_ids, segment_ids, input_mask, label_ids, exist_ids,all_aspects,pos,dis,all_pos_sent,all_noun_label)

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
