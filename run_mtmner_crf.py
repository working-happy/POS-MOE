from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from my_bert.mner_modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig, MTCCMBertForMMTokenClassificationCRF)
from my_bert.optimization import BertAdam, warmup_linear
from my_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

from torchvision import transforms
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support

from ner_evaluate import evaluate_each_class
from ner_evaluate import evaluate

from utils.data import Data

class embedding(nn.Module):
    """
    char lstm + word lstm + crf
    """
    def __init__(self, data):
        super(embedding, self).__init__()
        print('build MUL_LSTM_MCA network...')

        if data.pretrain_object_embedding is not None:
            self.object_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(data.pretrain_object_embedding),
                                                                  freeze=True)
            # object_embeddings   (94, 200)

        # self.object_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_object_embedding))
        else:
            self.object_embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(self.random_embedding(data.object_alphabet.size(), data.object_emb_dim)), freeze=True)

    def get_embedding(self, input):
        object_embs = self.object_embeddings(input)
        return object_embs

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


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


class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None, poslabel=None):
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
        self.img_id = img_id
        self.label = label
        self.auxlabel = auxlabel
        self.poslabel = poslabel


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, label_id, auxlabel_id, poslabel_id, object_id, added_input_mask_visual):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id
        self.poslabel_id = poslabel_id
        self.object_id = object_id
        self.added_input_mask_visual = added_input_mask_visual



def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []

    print("The number of samples: "+ str(len(data)))
    return data


def mmreadfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename, encoding='UTF-8-sig')
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label= []
    pos = []
    auxlabel = []
    imgid = ''
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1]+'.jpg'
            continue
        if line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label, pos))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                pos = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        pos.append(splits[2][:-1])
        cur_label = splits[1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) >0:
        data.append((sentence,label, pos))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []
        pos = []

    print("The number of samples: "+ str(len(data)))
    print("The number of images: "+ str(len(imgs)))
    return data, imgs, auxlabels


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
        return readfile(input_file)

    def _read_mmtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return mmreadfile(input_file)


class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "valid.txt"))
        return self._create_examples(data, imgs, auxlabels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def get_pos(self):
        return ["O", "NN", "JJ", "VB", "X", "[CLS]", "[SEP]"]

    def get_auxlabels(self):
        return ["O", "B", "I", "X", "[CLS]", "[SEP]"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[CLS]']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[SEP]']

    def _create_examples(self, lines, imgs, auxlabels, set_type):
        examples = []
        for i, (sentence, label, pos) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            pos = pos
            examples.append(MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel, poslabel=pos))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features


def convert_mm_examples_to_features(examples, label_list, auxlabel_list, max_seq_length, tokenizer, crop_size, path_img, poslabel_list, object_file, object_alphabet):
    """Loads a data file into a list of `InputBatch`s."""

    img2object = {}
    with open(object_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            imgID = line[0].split(".")[0]  # ID
            object = line[1:]
            img2object[imgID] = object

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}
    poslabel_map = {label: i for i, label in enumerate(poslabel_list, 1)}

    features = []
    count = 0

    for (ex_index, example) in enumerate(examples):
        object_Ids = []
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        poslabellist = example.poslabel
        tokens = []
        labels = []
        auxlabels = []
        poslabels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)     # 分词
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            poslabel_1 = poslabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                    poslabels.append(poslabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
                    poslabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
            poslabels = poslabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        poslabel_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        auxlabel_ids.append(auxlabel_map["[CLS]"])
        poslabel_ids.append(poslabel_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
            poslabel_ids.append(poslabel_map[poslabels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        auxlabel_ids.append(auxlabel_map["[SEP]"])
        poslabel_ids.append(poslabel_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)  # 1 or 49 is for encoding regional image representations
        added_input_mask_visual = [1] * (len(input_ids) + 4)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            added_input_mask.append(0)
            added_input_mask_visual.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            auxlabel_ids.append(0)
            poslabel_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length
        assert len(poslabel_ids) == max_seq_length

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)

        image_name = image_name.replace('.jpg', '')
        img_obj = img2object[image_name]

        for object in img_obj:
            object_Ids.append(object_alphabet.get_index(object))

        object_tensor = [0]*4
        object_lengths = len(object_Ids)  # 所有图像对象长度
        if object_lengths < 4:
            object_tensor[:object_lengths] = object_Ids
        else:
            object_tensor[:4] = object_Ids[:4]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))
            logger.info("poslabel: %s" % " ".join([str(x) for x in poslabel_ids]))

        features.append(
            MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                          segment_ids=segment_ids, label_id=label_ids, auxlabel_id=auxlabel_ids, poslabel_id=poslabel_ids, object_id = object_tensor, added_input_mask_visual=added_input_mask_visual))

    print('the number of problematic samples: ' + str(count))

    return features


def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",         # data/twitter2015
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,          # bert-base-cased
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",            # twitter2015
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",              # out/twitter2015_MTCCMBert_mtmner_crf_output/
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",        # 128
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",              # --do_train
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",               # --do_eval
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",      # 32
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=12.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=32,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--mm_model', default='MTCCMBert', help='model name')  # 'MTCCMBert', 'NMMTCCMBert'
    parser.add_argument('--layer_num1', type=int, default=1, help='number of txt2img layer')    # 1
    parser.add_argument('--layer_num2', type=int, default=1, help='number of img2txt layer')    # 1
    parser.add_argument('--layer_num3', type=int, default=1, help='number of txt2txt layer')    # 1
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--path_image', default='data/twitter2015_images/', help='path to images')
    parser.add_argument('--image_embedding', default='data/twitter2015_images_label.txt', help='path to images')
    #parser.add_argument('--mm_model', default='TomBert', help='model name') #
    args = parser.parse_args()

    logger.info('lr:{}'.format( args.learning_rate))

    if args.task_name == "twitter2017":
        args.path_image = "data/twitter2017_images/"
        args.image_embedding = "data/twitter2017_images_label.txt"
    elif args.task_name == "twitter2015":
        args.path_image = "data/twitter2015_images/"
        args.image_embedding = "data/twitter2015_images_label.txt"
    elif args.task_name == "test":
        args.path_image = "data/test_images/"
        args.image_embedding = "data/test_images_label.txt"
    
    processors = {
        "twitter2015": MNERProcessor,
        "twitter2017": MNERProcessor,
        "test": MNERProcessor,
        }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # device = torch.device("cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps   # 32

    #'''
    if args.task_name == "twitter2015":
        args.num_train_epochs = 50.0
    if args.task_name == "twitter2017":
        args.num_train_epochs = 40.0
    if args.task_name == "test":
        args.num_train_epochs = 24.0
    #'''

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        for i in os.listdir(args.output_dir):
            os.remove(args.output_dir + i)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()      # twitter2015

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    ##################  visual embedding####################################
    data = Data()
    object_alphabet = data.build_object_alphabet(args.image_embedding)
    data.build_pretrain_emb()
    ##################  visual embedding####################################

    processor = processors[task_name]()
    label_list = processor.get_labels()
    pos_list = processor.get_pos()
    auxlabel_list = processor.get_auxlabels()       # 辅助标签
    num_labels = len(label_list)+1 # label 0 corresponds to padding, label in label_list starts from 1   /13
    auxnum_labels = len(auxlabel_list)+1 # label 0 corresponds to padding, label in label_list starts from 1    /7
    posnum_labels = len(pos_list)+1
    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id()

    #''' initialization of our conversion matrix, in our implementation, it is a 7*12 matrix initialized as follows:
    trans_matrix = np.zeros((auxnum_labels,num_labels), dtype=float)     # 7*13
    trans_matrix[0,0]=1 # pad to pad
    trans_matrix[1,1]=1 # O to O
    trans_matrix[2,2]=0.25 # B to B-MISC
    trans_matrix[2,4]=0.25 # B to B-PER
    trans_matrix[2,6]=0.25 # B to B-ORG
    trans_matrix[2,8]=0.25 # B to B-LOC
    trans_matrix[3,3]=0.25 # I to I-MISC
    trans_matrix[3,5]=0.25 # I to I-PER
    trans_matrix[3,7]=0.25 # I to I-ORG
    trans_matrix[3,9]=0.25 # I to I-LOC
    trans_matrix[4,10]=1   # X to X
    trans_matrix[5,11]=1   # [CLS] to [CLS]
    trans_matrix[6,12]=1   # [SEP] to [SEP]
    '''
    trans_matrix = np.zeros((num_labels, auxnum_labels), dtype=float)
    trans_matrix[0,0]=1 # pad to pad
    trans_matrix[1,1]=1
    trans_matrix[2,2]=1
    trans_matrix[4,2]=1
    trans_matrix[6,2]=1
    trans_matrix[8,2]=1
    trans_matrix[3,3]=1
    trans_matrix[5,3]=1
    trans_matrix[7,3]=1
    trans_matrix[9,3]=1
    trans_matrix[10,4]=1
    trans_matrix[11,5]=1
    trans_matrix[12,6]=1
    '''

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case) # 从网上获取模型

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

    if args.mm_model == 'MTCCMBert':
        model = MTCCMBertForMMTokenClassificationCRF.from_pretrained(args.bert_model,
              cache_dir=cache_dir, layer_num1=args.layer_num1, layer_num2=args.layer_num2, layer_num3=args.layer_num3,
              num_labels = num_labels, auxnum_labels = auxnum_labels)
    else:
        print('please define your MNER Model')

    net = getattr(resnet, 'resnet152')()     # 得到resnet152的值
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    if args.fp16:
        model.half()     # half()方法将模型中的float32转化为float16
    model.to(device)            # 模型加载GPU
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)         # 模型分布式计算
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    img_modal = embedding(data)
    if args.fp16:
        img_modal.half()     # half()方法将模型中的float32转化为float16
    img_modal.to(device)

    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

    if args.task_name != "test":
        open('./F1_Dev_{}'.format(args.task_name), 'w')
        open('./F1_Test_{}'.format(args.task_name), 'w')

    if args.do_train:
        # 训练集
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image, pos_list, args.image_embedding, object_alphabet)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in train_features], dtype=torch.long)
        all_poslabel_ids = torch.tensor([f.poslabel_id for f in train_features], dtype=torch.long)
        all_object_ids = torch.tensor([f.object_id for f in train_features], dtype=torch.long)
        all_added_input_mask_visual = torch.tensor([f.added_input_mask_visual for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, \
                     all_segment_ids, all_label_ids, all_auxlabel_ids, all_poslabel_ids, all_object_ids, all_added_input_mask_visual)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # 验证集
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image, pos_list, args.image_embedding, object_alphabet)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        all_poslabel_ids = torch.tensor([f.poslabel_id for f in eval_features], dtype=torch.long)
        all_object_ids = torch.tensor([f.object_id for f in eval_features], dtype=torch.long)
        all_added_input_mask_visual = torch.tensor([f.added_input_mask_visual for f in eval_features],
                                                   dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, \
                                  all_segment_ids, all_label_ids, all_auxlabel_ids, all_poslabel_ids, all_object_ids, all_added_input_mask_visual)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # 测试集
        test_eval_examples = processor.get_test_examples(args.data_dir)
        test_eval_features = convert_mm_examples_to_features(
            test_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image, pos_list, args.image_embedding, object_alphabet)
        all_input_ids = torch.tensor([f.input_ids for f in test_eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in test_eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in test_eval_features], dtype=torch.long)
        all_poslabel_ids = torch.tensor([f.poslabel_id for f in test_eval_features], dtype=torch.long)
        all_object_ids = torch.tensor([f.object_id for f in test_eval_features], dtype=torch.long)
        all_added_input_mask_visual = torch.tensor([f.added_input_mask_visual for f in test_eval_features],
                                                   dtype=torch.long)
        test_eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                  all_label_ids, all_auxlabel_ids, all_poslabel_ids, all_object_ids, all_added_input_mask_visual)
        # Run prediction for full data
        test_eval_sampler = SequentialSampler(test_eval_data)
        test_eval_dataloader = DataLoader(test_eval_data, sampler=test_eval_sampler, batch_size=args.eval_batch_size)

        max_dev_f1 = 0.0
        max_test_f1 = 0.0
        best_dev_epoch = 0
        best_test_epoch = 0
        logger.info("***** Running training *****")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            model.train()       # 作用是启用batch normalization和drop out
            img_modal.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)    # 将列表转换为元组
                # img_feats (8, 3, 224, 224)
                # object_ids (8, 4)
                input_ids, input_mask, added_input_mask, segment_ids, label_ids, auxlabel_ids, poslabel_ids, object_ids, added_input_mask_visual = batch
                with torch.no_grad():
                    img_emb = img_modal.get_embedding(object_ids)
                    # img_emb  (8, 4, 200)
                #img_att  [8, 2048, 7, 7]
                trans_matrix = torch.tensor(trans_matrix).to(device)
                neg_log_likelihood = model(poslabel_ids, input_ids, segment_ids, input_mask, added_input_mask,
                                                              trans_matrix, img_emb, added_input_mask_visual, label_ids, auxlabel_ids)
                if n_gpu > 1:
                    neg_log_likelihood = neg_log_likelihood.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(neg_log_likelihood)
                else:
                    neg_log_likelihood.backward()

                tr_loss += neg_log_likelihood.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            img_modal.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            label_map[0] = "PAD"
            for input_ids, input_mask, added_input_mask, segment_ids, label_ids, auxlabel_ids, poslabel_ids, object_ids, added_input_mask_visual in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                auxlabel_ids = auxlabel_ids.to(device)
                poslabel_ids = poslabel_ids.to(device)
                object_ids = object_ids.to(device)
                added_input_mask_visual = added_input_mask_visual.to(device)
                #trans_matrix = torch.tensor(trans_matrix).to(device)

                with torch.no_grad():
                    img_emb = img_modal.get_embedding(object_ids)
                    predicted_label_seq_ids = model(poslabel_ids, input_ids, segment_ids, input_mask, added_input_mask, trans_matrix, img_emb, added_input_mask_visual)

                #logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                #logits = logits.detach().cpu().numpy()
                # logits = predicted_label_seq_ids.detach().cpu().numpy()
                logits = predicted_label_seq_ids
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            #temp_1.pop()
                            #temp_2.pop()
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("***** Dev Eval results *****")
            logger.info("\n%s", report)
            #eval_true_label = np.concatenate(y_true)
            #eval_pred_label = np.concatenate(y_pred)
            #precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
            F_score_dev = float(report.split('\n')[-4].split('      ')[-2].split('    ')[-1])
            print("F-score: ", F_score_dev)

            if args.task_name != "test":
                F1_Dev = open('./F1_Dev_{}'.format(args.task_name), 'a')
                F1_Dev.write(str(F_score_dev) + '\t')
                F1_Dev.close()

            logger.info("***** Running Test evaluation *****")
            logger.info("  Num examples = %d", len(test_eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            label_map[0] = "PAD"
            for input_ids, input_mask, added_input_mask, segment_ids, label_ids, auxlabel_ids, poslabel_ids, object_ids, added_input_mask_visual in test_eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                auxlabel_ids = auxlabel_ids.to(device)
                poslabel_ids = poslabel_ids.to(device)
                object_ids = object_ids.to(device)
                added_input_mask_visual = added_input_mask_visual.to(device)
                #trans_matrix = torch.tensor(trans_matrix).to(device)

                with torch.no_grad():
                    img_emb = img_modal.get_embedding(object_ids)
                    predicted_label_seq_ids = model(poslabel_ids, input_ids, segment_ids, input_mask, added_input_mask, trans_matrix, img_emb, added_input_mask_visual)

                # logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                # logits = logits.detach().cpu().numpy()
                # logits = predicted_label_seq_ids.detach().cpu().numpy()
                logits = predicted_label_seq_ids
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            # temp_1.pop()
                            # temp_2.pop()
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("***** Test Eval results *****")
            logger.info("\n%s", report)
            F_score_test = float(report.split('\n')[-4].split('      ')[-2].split('    ')[-1])

            if args.task_name != "test":
                F1_Test = open('./F1_Test_{}'.format(args.task_name), 'a')
                F1_Test.write(str(F_score_test) + '\t')
                F1_Test.close()

            if F_score_dev > max_dev_f1:
                # Save a trained model and the associated configuration
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                encoder_to_save = img_modal.module if hasattr(img_modal,
                                                            'module') else img_modal  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                "label_map": label_map}
                json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
                max_dev_f1 = F_score_dev
                best_dev_epoch = train_idx
            if F_score_test > max_test_f1:
                max_test_f1 = F_score_test
                best_test_epoch = train_idx

    print("**************************************************")
    print("The best epoch on the dev set: ", best_dev_epoch)
    print("The best Micro-F1 score on the dev set: ", max_dev_f1)
    print("The best epoch on the test set: ", best_test_epoch)
    print("The best Micro-F1 score on the test set: ", max_test_f1)
    print('\n')

    config = BertConfig(output_config_file)
    if args.mm_model == 'MTCCMBert':
        model = MTCCMBertForMMTokenClassificationCRF(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2,
                                            layer_num3=args.layer_num3, num_labels=num_labels, auxnum_labels = auxnum_labels)
    else:
        print('please define your MNER Model')

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    encoder_state_dict = torch.load(output_encoder_file)
    img_modal.load_state_dict(encoder_state_dict)
    img_modal.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image, pos_list, args.image_embedding, object_alphabet)
        logger.info("***** Running Test Evaluation with the Best Model on the Dev Set*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        all_poslabel_ids = torch.tensor([f.poslabel_id for f in eval_features], dtype=torch.long)
        all_object_ids = torch.tensor([f.object_id for f in eval_features], dtype=torch.long)
        all_added_input_mask_visual = torch.tensor([f.added_input_mask_visual for f in eval_features],
                                                   dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                  all_label_ids, all_auxlabel_ids, all_poslabel_ids, all_object_ids, all_added_input_mask_visual)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        img_modal.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        label_map[0] = "PAD"
        for input_ids, input_mask, added_input_mask, segment_ids, label_ids, auxlabel_ids, poslabel_ids, object_ids, added_input_mask_visual in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            auxlabel_ids = auxlabel_ids.to(device)
            poslabel_ids = poslabel_ids.to(device)
            trans_matrix = torch.tensor(trans_matrix).to(device)
            object_ids = object_ids.to(device)
            added_input_mask_visual = added_input_mask_visual.to(device)

            with torch.no_grad():
                img_emb = img_modal.get_embedding(object_ids)
                predicted_label_seq_ids  = model(poslabel_ids, input_ids, segment_ids, input_mask, added_input_mask, trans_matrix, img_emb, added_input_mask_visual)
            
            #logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            #logits = logits.detach().cpu().numpy()
            # logits = predicted_label_seq_ids.detach().cpu().numpy()
            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i,mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []
                #count = 0
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])
                        #if label_map[label_ids[i][j]].startswith("B"):
                            #count += 1
                    else:
                        #temp_1.pop()
                        #temp_2.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)
                #if count > 1:
                    #multi_ent_y_true.append(temp_1)
                    #multi_ent_y_pred.append(temp_2)

        report = classification_report(y_true, y_pred,digits=4)
        #multi_ent_report = classification_report(multi_ent_y_true, multi_ent_y_pred,digits=4)

        sentence_list = []
        test_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))
        output_pred_file = os.path.join(args.output_dir, "mtmner_pred.txt")
        fout = open(output_pred_file, 'w')
        for i in range(len(y_pred)):
            sentence = test_data[i][0]
            sentence_list.append(sentence)
            img = imgs[i]
            samp_pred_label = y_pred[i]
            samp_true_label = y_true[i]
            fout.write(img+'\n')
            fout.write(' '.join(sentence)+'\n')
            fout.write(' '.join(samp_pred_label)+'\n')
            fout.write(' '.join(samp_true_label)+'\n'+'\n')
        fout.close()

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
        print("Overall: ", p, r, f1)
        per_f1, per_p, per_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'PER')
        print("Person: ", per_p, per_r, per_f1)
        loc_f1, loc_p, loc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'LOC')
        print("Location: ", loc_p, loc_r, loc_f1)
        org_f1, org_p, org_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'ORG')
        print("Organization: ", org_p, org_r, org_f1)
        misc_f1, misc_p, misc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'MISC')
        print("Miscellaneous: ", misc_p, misc_r, misc_f1)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)
            writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
            writer.write("Person: " + str(per_p) + ' ' + str(per_r) + ' ' + str(per_f1) + '\n')
            writer.write("Location: " + str(loc_p) + ' ' + str(loc_r) + ' ' + str(loc_f1) + '\n')
            writer.write("Organization: " + str(org_p) + ' ' + str(org_r) + ' ' + str(org_f1) + '\n')
            writer.write("Miscellaneous: " + str(misc_p) + ' ' + str(misc_r) + ' ' + str(misc_f1) + '\n')

if __name__ == "__main__":
    main()
