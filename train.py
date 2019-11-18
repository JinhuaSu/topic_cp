#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch
from pytorch_pretrained_bert import BertConfig

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer, Decoder, get_initial_s
from models.trainer import build_trainer
from others.logging import logger, init_logger
# from pytorch_pretrained_bert import BertTokenizer
from stanfordcorenlp import StanfordCoreNLP
from models.tokenization import BertTokenizer



model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
            device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):

    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])
    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks, args.init_method)
        print('gpu_rank %d' %gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        recover_all = False
        pt=None
        if args.recover_from != '':
            pt = args.recover_from
            recover_all = True
        elif args.extract_model != '':
            pt = args.extract_model

        if args.mode == 'abs':
            abs_train(args, device_id, pt, recover_all)
        else:
            train(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def wait_and_validate(args, device_id):

    timestep = 0
    print('test_all', args.test_all)
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = memory_validate(args,  device_id, cp)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            memory_test(args,  device_id, cp)
        # for i, cp in enumerate(cp_files):
        #     step = int(cp.split('.')[-2].split('_')[-1])
        #     test(args,  device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args,  device_id, cp, step)
                    test(args,  device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args,  device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    valid_iter =data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test(args, device_id, pt, step):

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter,step)


def memory_validate(args, device_id, pt):
    step = int(pt.split('.')[-2].split('_')[-1])

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.key_validate(test_iter,step)
    return stats.xent()


def memory_test(args, device_id, pt):
    step = int(pt.split('.')[-2].split('_')[-1])

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.key_test(test_iter,step)


def baseline(args, cal_lead=False, cal_oracle=False):

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)

    trainer = build_trainer(args, device_id, None, None)
    #
    if (cal_lead):
        trainer.test(test_iter, 0, cal_lead=True)
    elif (cal_oracle):
        trainer.test(test_iter, 0, cal_oracle=True)


def abs_decoder(args, device_id, pt):

    step = int(pt.split('.')[-2].split('_')[-1])

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)

    # decoder
    decoder = Decoder(model.bert.model.config.hidden_size//2, model.bert.model.config.vocab_size,
                      model.bert.model.config.hidden_size, model.bert.model.embeddings, device, logger)  # 2*hidden_dim = embedding_size


    # get initial s_t
    s_t_1 = get_initial_s(model.bert.model.config.hidden_size, device)

    model.load_cp(checkpoint)
    s_t_1.load_cp(checkpoint)
    decoder.load_cp(checkpoint)

    model.eval()
    decoder.eval()
    s_t_1.eval()
    # tokenizer,nlp
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                              never_split=('[SEP]','[CLS]','[PAD]','[unused0]','[unused1]','[unused2]','[UNK]'),
                                              no_word_piece=True)
    nlp = StanfordCoreNLP(r'/home1/bqw/stanford-corenlp-full-2018-10-05')
    # nlp.logging_level = 10

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None, decoder=decoder, get_s_t= s_t_1, device=device_id,
                            tokenizer=tokenizer, nlp=nlp, extract_num=args.extract_num)
    trainer.abs_decode(test_iter,step)

def abs_train(args, device_id, pt, recover_all=False):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # load extractive model
    if pt != None:
        test_from = pt
        logger.info('Loading checkpoint from %s' % test_from)
        checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        print(args)
    config = BertConfig.from_json_file(args.bert_config_path)
    # build extractive model
    model = Summarizer(args, device_id, load_pretrained_bert=False, bert_config=config)

    # decoder
    decoder = Decoder(model.bert.model.config.hidden_size//2, model.bert.model.config.vocab_size,
                      model.bert.model.config.hidden_size, model.bert.model.embeddings, device_id)  # 2*hidden_dim = embedding_size

    # get initial s_t
    s_t_1 = get_initial_s(model.bert.model.config.hidden_size, device_id)
    if recover_all:
        model.load_cp(checkpoint)
        s_t_1.load_cp(checkpoint)
        decoder.load_cp(checkpoint)
        optim = model_builder.build_optim(args, [model, decoder, s_t_1], checkpoint)

    elif pt != None:
        model.load_cp(checkpoint)
        optim = model_builder.build_optim(args, [model, decoder, s_t_1], checkpoint)
    else:
        optim = model_builder.build_optim(args, [model, decoder, s_t_1], None)


    # tokenizer,nlp
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                              never_split=('[SEP]','[CLS]','[PAD]','[unused0]','[unused1]','[unused2]','[UNK]'),
                                              no_word_piece=True)
    nlp = StanfordCoreNLP(r'/home1/bqw/stanford-corenlp-full-2018-10-05')

    # build optim

    # load train dataset
    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device_id,
                                                 shuffle=True, is_test=False)

    # build trainer
    trainer = build_trainer(args, device_id, model, optim, decoder=decoder, get_s_t= s_t_1, device=device_id,
                            tokenizer=tokenizer, nlp=nlp)
    trainer.abs_train(train_iter_fct, args.train_steps)


def train(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)


    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                                 shuffle=True, is_test=False)

    model = Summarizer(args, device, load_pretrained_bert=True)
    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        model.load_cp(checkpoint)
        optim = model_builder.build_optim(args, model, checkpoint)
    else:
        optim = model_builder.build_optim(args, model, None)

    logger.info(model)
    trainer = build_trainer(args, device_id, model, optim, memo_pre=args.memo_pre, normal_pre=args.normal_pre)
    trainer.train(train_iter_fct, args.train_steps)


def oracle(args, device_id, pt):

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, None, None)

    print('___________________________________________________________________________')

    trainer.oracle(test_iter,cal_lead=False, cal_oracle=False,each=True)


def cluster_valid_test(args, device_id):
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    rouge_lst = []
    for i, cp in enumerate(cp_files):
        step = int(cp.split('.')[-2].split('_')[-1])
        rouges = cluster_valid(args, device_id, cp, step)
        rouge_lst.append((rouges, cp))
        max_step = rouge_lst.index(max(rouge_lst))
        if (i - max_step > 10):
            break
    rouge_lst = sorted(rouge_lst, key=lambda x: x[0], reverse=True)[:3]
    logger.info('PPL %s' % str(rouge_lst))
    for rouge, cp in rouge_lst:
        step = int(cp.split('.')[-2].split('_')[-1])
        cluster_test(args, device_id, cp, step)

def cluster_test(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.cluster_test(test_iter,step)


def cluster_valid(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    rouges=trainer.cluster_test(test_iter,step)
    return rouges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-encoder", default='classifier', type=str, choices=['classifier','transformer','rnn','baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train','validate','test','cluster','oracle','baseline',
                                                                     'abs', 'decoder', 'memory_test'])
    parser.add_argument("-bert_data_path", default='../bert_data/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-bert_config_path", default='../bert_config_uncased_base.json')
    parser.add_argument("-recover_from", default='')
    parser.add_argument("-extract_model", default='')


    parser.add_argument('-init_method', default='tcp://localhost:10000')


    parser.add_argument("-batch_size", default=1000, type=int)

    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=512, type=int)
    parser.add_argument("-heads", default=4, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)
    parser.add_argument("-rnn_size", default=512, type=int)
    parser.add_argument("-memo_pre", default=10000, type=int)
    parser.add_argument("-normal_pre", default=1000, type=int)


    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='', type=str)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-extract_num", default=5, type=int)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)



    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-decoder_model', default='../models/bert_transformer/model_step_36000.pt')

    parser.add_argument("-cluster_result_path", default='../cluster')


    parser.add_argument("--local_rank")


    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    if(args.world_size>1):
        multi_main(args)
    elif (args.mode == 'train'):
        train(args, device_id)
    elif (args.mode == 'validate'):
        wait_and_validate(args, device_id)
    elif (args.mode == 'baseline'):
        baseline(args, cal_lead=False, cal_oracle=True)
    elif (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test(args, device_id, cp, step)
    elif (args.mode == 'cluster_test'):
        cp='/home1/bqw/sum/BertSum/models/bert_transformer/model_step_26000.pt'
        cluster_test(args, device_id, cp, 26000)
    elif (args.mode == 'cluster'):
        cluster_valid_test(args, device_id)
    elif (args.mode == 'oracle'):
        print('_______________________________________________________')
        cp = args.test_from
        oracle(args,device_id, cp)
    elif (args.mode == 'abs'):
        recover_all = False
        if args.recover_from != '':
            pt = args.recover_from
            recover_all = True
        else:
            pt = args.extract_model
        # pt = '/home1/bqw/sum/BertSum/models/bert_transformer/model_step_18000.pt'
        # pt = '/home1/bqw/sum/BertSum/models/bert_transformer/model_step_18000.pt'
        abs_train(args, device_id, pt, recover_all)
    elif (args.mode == 'memory_test'):
        memory_test(args, device_id, args.test_from)

    elif (args.mode == 'decoder'):
        pt = args.decoder_model
        abs_decoder(args, device_id, pt)


