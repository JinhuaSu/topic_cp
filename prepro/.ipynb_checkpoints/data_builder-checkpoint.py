import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from os.path import join as pjoin

import torch
from multiprocess import Pool
# from pytorch_pretrained_bert import BertTokenizer
from models.tokenization import BertTokenizer
from others.logging import logger
from others.utils import clean
from prepro.utils import _get_word_ngrams
# from stanfordcorenlp import StanfordCoreNLP
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS



def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    # pos_tag = []
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        # _pos_tag = [t['pos'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if (flag):
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)
            # pos_tag.append(_pos_tag)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    # assert len(' '.join([' '.join(i) for i in pos_tag]).split()) == len(' '.join([' '.join(i) for i in source]).split())
    return source, tgt


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size=3):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def each_selection(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    # abstract = _rouge_clean(' '.join(abstract)).split()
    # sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    # evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    # reference_1grams = _get_word_ngrams(1, [abstract])
    # evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    # reference_2grams = _get_word_ngrams(2, [abstract])

    abstract = sum(abstract_sent_list, [])
    abstracts = [_rouge_clean(' '.join(s)).split() for s in abstract]
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = [_get_word_ngrams(1, [sent]) for sent in abstracts]
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = [_get_word_ngrams(2, [sent]) for sent in abstracts]

    selected = []
    for _abstract in range(len(abstracts)):
        rouge_score = []
        reference_1 = set(reference_1grams[_abstract])
        reference_2 = set(reference_2grams[_abstract])
        # print('------------------------------------------------------------')
        # print(len(sents))
        # print(len(abstracts))
        for _sent in range(len(sents)):
            if _sent in rouge_score:
                continue
            candidates_1 = set(evaluated_1grams[_sent])
            candidates_2 = set(evaluated_2grams[_sent])
            rouge_1 = cal_rouge(candidates_1, reference_1)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2)['f']
            _rouge_score = rouge_1 + rouge_2
            # print(_rouge_score)
            rouge_score.append(_rouge_score)
        # print('length',len(rouge_score))
        if len(rouge_score) > 0:
            cur_index = rouge_score.index(max(rouge_score))
            if cur_index not in selected:
                selected.append(cur_index)
    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                                       never_split=('[SEP]','[CLS]','[PAD]','[unused0]','[unused1]','[unused2]','[UNK]'),
                                                       no_word_piece=True)

        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']
        # self.pos_dict = get_pos_dic()

    def preprocess(self, src, tgt, oracle_ids, lda_model_tfidf, lda_dict, topic_num=10):
        # lda_model_tfidf = models.ldamodel.LdaModel.load('/home1/bqw/sum/sum_topic_10.model')
        # lda_dict = corpora.Dictionary.load('/home1/bqw/sum/topic.dict')
        # corpus = corpora.MmCorpus('/home1/bqw/sum/topic.mm')
        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        src = src[:self.args.max_nsents]
        # labels = [labels[i] for i in idxs]
        # labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        ##############################
        # topic label
        topic_src = sum(src, [])
        topic_src = [i for i in topic_src if i not in STOPWORDS][:510]
        bow_vector = lda_dict.doc2bow(topic_src)
        topic_pro = lda_model_tfidf[bow_vector]
        back = [(i, 0.0) for i in range(topic_num)]
        have_set = [i[0] for i in topic_pro]
        for i, tup in enumerate(back):
            if tup[0] not in have_set:
                topic_pro.append(tup)
        topic_pro.sort(key=lambda i: i[0])
        topic_pro = [i[1] for i in topic_pro]
        ################################
        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        if len(src_subtokens)>510:
            if src_subtokens[510] == '[CLS]':
                src_subtokens = src_subtokens[:509]
            elif src_subtokens[509] == '[CLS]':
                src_subtokens =src_subtokens[:508]
            else:
                src_subtokens = src_subtokens[:510]

        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        select_src = []
        sub_list = []
        for i in src_subtokens:
            if i=='[CLS]':
                continue
            elif i=='[SEP]':
                select_src.append(sub_list)
                sub_list=[]
            else:
                sub_list.append(i)

        assert len(select_src)== src_subtokens.count('[CLS]')
        oracle_ids = greedy_selection(select_src, tgt, 3)
        # oracle_ids = combination_selection(select_src, tgt, 3)

        labels = [0] * len(select_src)
        if len(oracle_ids)<3:
            return None
        # print(oracle_ids)
        # print('select_src',len(select_src))
        for l in oracle_ids:
            labels[l] = 1
        # print(sum(labels))
        assert sum(labels) == 3

        # tgt
        tgt_text = ' [unused2] '.join([' '.join(tt) for tt in tgt])
        tgt_subtokens_source = self.tokenizer.tokenize(tgt_text)
        tgt_subtokens = tgt_subtokens_source + ['[unused1]', '[PAD]']
        # tgt_subtokens = tgt_subtokens_source + ['[unused1]']

        tgt_subtokens = tgt_subtokens[:200]
        tgt_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtokens)
        # use [unused0] replace [start]
        # [unused2] replace [<q>]
        # [unused1] replace [end]

        decoder_tgt = ['[unused0]'] + tgt_subtokens_source + ['[unused1]']
        # decoder_tgt = ['[unused0]'] + tgt_subtokens_source

        decoder_tgt = decoder_tgt[:200]
        decoder_tgt_idxs = self.tokenizer.convert_tokens_to_ids(decoder_tgt)


        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = ' <q> '.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt, tgt_idxs, decoder_tgt_idxs, topic_pro


def format_to_bert(args):
    lda_model_tfidf = models.ldamodel.LdaModel.load('/home1/bqw/sum/sum_topic_10.model')
    lda_dict = corpora.Dictionary.load('/home1/bqw/sum/topic.dict')
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']


    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')), lda_model_tfidf,
                          lda_dict))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):

    json_file, args, save_file, lda_model_tfidf, lda_dict = params

    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        oracle_ids = None
        # if (args.oracle_mode == 'greedy'):
        #     oracle_ids = greedy_selection(source, tgt)
        # elif (args.oracle_mode == 'combination'):
        #     oracle_ids = combination_selection(source, tgt, 3)
        # elif (args.oracle_mode == 'each'):
        #     oracle_ids = each_selection(source, tgt)
        b_data = bert.preprocess(source, tgt, oracle_ids, lda_model_tfidf, lda_dict)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, tgt_idxs, decoder_tgt_idxs, topic_pro = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "tgt_idxs": tgt_idxs,
                       "decoder_tgt_idxs": decoder_tgt_idxs, "topic_pro": topic_pro}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
# src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt, tgt_idxs, decoder_tgt_idxs


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    # attention backward
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP' ,'-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}


def get_pos_dic():
    POS_TAG = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
               'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '[CLS]', '[SEP]']
    pos_dict = dict(zip(POS_TAG, list(range(len(POS_TAG)))))
    return pos_dict

