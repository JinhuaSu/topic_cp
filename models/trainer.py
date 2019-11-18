import os

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as func
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import distributed
# import onmt
from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str

from sklearn.cluster import KMeans
from sklearn import cluster
from multiprocessing import Pool
from rouge import Rouge
import numpy as np
import torchsnooper
from models.decoder import BeamSearch
from torch import autograd
from time import time
from models.model_builder import Memory, pair_loss


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


# decoder=None, get_s_t=None, device=None, tokenizer=None,nlp=None
def build_trainer(args, device_id, model,
                  optim, decoder=None, get_s_t=None, device=None, tokenizer=None,nlp=None,extract_num=5,memo_pre=10000,
                  normal_pre=1000):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    # device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)
    POS_TAG = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
               'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '[CLS]', '[SEP]']
    pos_dict = dict(zip(POS_TAG, list(range(len(POS_TAG)))))
    
    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager, decoder,
                      get_s_t, device, tokenizer, nlp, pos_dict, extract_num=extract_num, memo_pre=memo_pre,
                      normal_pre=normal_pre)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


def get_position(real_heap):
    pos = []
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(real_heap)
    core_vec = kmeans.cluster_centers_
    for vec in core_vec:
        dis_vec = np.zeros(np.shape(real_heap)[0])
        # print('dis_vec',dis_vec)
        # print(np.shape(dis_vec))
        for i, row in enumerate(real_heap):
            dis_vec[i]=-np.dot(row, vec) / (np.linalg.norm(row) * np.linalg.norm(vec))
            # dis_tmp = sum((row - vec) ** 2)
            # dis_vec[i] = dis_tmp
        rank_vec = np.argsort(dis_vec)
        # print('rank_vec',rank_vec)
        # print(np.shape(rank_vec))
        if len(pos) == 0:
            pos.append(rank_vec[0])
        else:
            for i in range(len(rank_vec)):
                if rank_vec[i] in pos:
                    continue
                else:
                    pos.append(rank_vec[i])
                    break
    pos.sort()
    return pos


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model,  optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None, decoder=None, get_s_t=None, device=None, tokenizer=None, nlp=None, pos_dict=None,
                 extract_num=5, memo_pre=10000, normal_pre=1000):
        # Basic attributes.
        self.args = args
        self.tokenizer = tokenizer
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.model = model
        self.decoder = decoder
        self.get_s_t = get_s_t
        self.device = device
        self.nlp = nlp
        self.pos_dict = pos_dict
        self.use_pos_tag = False
        self.count = 0
        self.interval = False
        self.extract_num = extract_num
        self.memo_num = 0
        self.normal_num = 0
        self.memo_pre=memo_pre
        self.normal_pre=normal_pre


        self.memo = False
        self.norm = True


        self.eps = 1e-7
        self.loss = torch.nn.BCELoss(reduction='none')
        self.pair_loss = pair_loss()
        self.loss_per = 0.9

        self._error = 0
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)

                    normalization += batch.batch_size
                    accum += 1

                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation_topic(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._train_save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()
        return total_stats

    def _gradient_accumulation_topic(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls
            batch_size = src.size(0)
            
            topic_label = batch.topic_pro
            ################################################### topic predict
            top_vec = self.model.bert(src, segs, mask)
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            # b x n x m
            sents_vec = sents_vec * mask_cls[:, :, None].float()
            
            #############################################################################
            # topic
            doc_vec = self.model.doc_extractor(sents_vec, mask_cls)
            pre_label = self.model.topic_predictor(doc_vec).squeeze(-1)
            topic_loss = self.loss(pre_label, topic_label.float())
            topic_loss = topic_loss.sum()
            ##############
            # version_2

            # b x m x 1
            topic_vec = topic_label.mm(self.model.topic_embedding).unsqueeze(-1)
            # b x n
            match_score = torch.matmul(sents_vec, doc_vec.unsqueeze(-1)).squeeze(-1)
            sents_num = match_score.size(1)
            sent_scores = F.softmax(match_score, dim=1)

            # hinge loss
            positive_score = sent_scores[labels.type(torch.uint8)].view(batch_size, -1, 1).expand((batch_size, -1, sents_num))
            back_score = sent_scores.unsqueeze(1).expand((batch_size, 3, sents_num))
            final_score = back_score - positive_score
            final_score = torch.where(final_score < 0, torch.zeros_like(final_score), final_score)
            
            unseen = labels.view((batch_size, 1, sents_num)).expand((batch_size, 3, sents_num)).type(torch.uint8)
            unseen = ~unseen
            final_score = (final_score * unseen.float()).sum(1)
            
            hinge_loss = (final_score * mask_cls.float()).sum()
            (hinge_loss / hinge_loss.numel() + topic_loss / topic_loss.numel()).backward()

            # version_1
            # topic vector

            # topic_vec = topic_label.mm(self.model.topic_embedding).unsqueeze(1)
            # sents_vec = self.model.memory(sents_vec, topic_vec)
            # sent_scores = self.model.encoder(sents_vec, mask_cls).squeeze(-1)
            #
            # loss = self.loss(sent_scores, labels.float())
            # loss = (loss * mask_cls.float()).sum()
            # (loss / loss.numel() + topic_loss / topic_loss.numel()).backward()
            ###########################################################################
            # # key_memory
            # sents_vec = self.model.key_memory(sents_vec, self.model.topic_word_emb)
            # sent_scores = self.model.encoder(sents_vec, mask_cls).squeeze(-1)
            # # loss
            # loss = self.loss(sent_scores, labels.float())
            # loss = (loss * mask_cls.float()).sum()
            # (loss / loss.numel()).backward()
            ###########################################################################
            batch_stats = Statistics(float(hinge_loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def key_validate(self, valid_iter, step):
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls
                topic_label = batch.topic_pro
                ################################################### topic predict
                top_vec = self.model.bert(src, segs, mask)
                sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
                sents_vec = sents_vec * mask_cls[:, :, None].float()
                # key_memory
                doc_vec = self.model.doc_extractor(sents_vec, mask_cls)
                match_score = torch.matmul(sents_vec, doc_vec.unsqueeze(-1)).squeeze(-1)
                sents_num = match_score.size(1)
                sent_scores = F.softmax(match_score, dim=1)

                # loss
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask_cls.float()).sum()

                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def key_test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        print('memory test')
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        self.count+=1
                        if self.count > 30000:
                            break
                        src = batch.src
                        labels = batch.labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask
                        mask_cls = batch.mask_cls
                        topic_label = batch.topic_pro


                        gold = []
                        pred = []

                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            ################################################### topic predict
                            top_vec = self.model.bert(src, segs, mask)
                            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
                            sents_vec = sents_vec * mask_cls[:, :, None].float()
                            # key_memory
                            doc_vec = self.model.doc_extractor(sents_vec, mask_cls)
                            match_score = torch.matmul(sents_vec, doc_vec.unsqueeze(-1)).squeeze(-1)
                            sents_num = match_score.size(1)
                            sent_scores = F.softmax(match_score, dim=1)

                            #####################################################
                            sent_scores = sent_scores + mask_cls.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                        # selected_ids = np.sort(selected_ids,1)
                        # selected_ids: torch.Size([5, 17])
                        # select the top 3
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if (len(batch.src_str[i]) == 0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if (j >= len(batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if (self.args.block_trigram):
                                    if (not _block_tri(candidate, _pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if (self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip() + '\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip() + '\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)
        return stats

    def topic_validate(self, valid_iter, step):
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls
                topic_label = batch.topic_pro

                ################################################### topic predict
                top_vec = self.model.bert(src, segs, mask)
                sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
                sents_vec = sents_vec * mask_cls[:, :, None].float()
                pre_label = self.model.topic_predictor(sents_vec).squeeze(-1)
                # loss
                # topic_loss = self.loss(pre_label, topic_label.float())
                # topic_loss = topic_loss.sum()
                # topic vector
                topic_vec = pre_label.mm(self.model.topic_embedding).unsqueeze(1)
                # interactive
                match_score = torch.matmul(sents_vec, topic_vec).squeeze(-1)
                sent_scores = F.softmax(match_score, dim=1)
                # loss
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask_cls.float()).sum()

                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def topic_test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        print('memory test')
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        self.count+=1
                        if self.count > 50000:
                            break
                        src = batch.src
                        labels = batch.labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask
                        mask_cls = batch.mask_cls
                        topic_label = batch.topic_pro


                        gold = []
                        pred = []

                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            ################################################### topic predict
                            top_vec = self.model.bert(src, segs, mask)
                            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
                            sents_vec = sents_vec * mask_cls[:, :, None].float()
                            pre_label = self.model.topic_predictor(sents_vec).squeeze(-1)
                            # loss
                            # topic_loss = self.loss(pre_label, topic_label.float())
                            # topic_loss = topic_loss.sum()
                            # topic vector

                            topic_vec = pre_label.mm(self.model.topic_embedding).unsqueeze(1)
                            # interactive
                            match_score = torch.matmul(sents_vec, topic_vec).squeeze(-1)
                            sent_scores = F.softmax(match_score, dim=1)

                            sent_scores = sent_scores + mask_cls.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                        # selected_ids = np.sort(selected_ids,1)
                        # selected_ids: torch.Size([5, 17])
                        # select the top 3
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if (len(batch.src_str[i]) == 0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if (j >= len(batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if (self.args.block_trigram):
                                    if (not _block_tri(candidate, _pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if (self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip() + '\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip() + '\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)
        return stats

    def memory_validate(self, valid_iter, step):
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls


                ####################################################
                top_vec = self.model.bert(src, segs, mask)
                sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
                sents_vec = sents_vec * mask_cls[:, :, None].float()

                memo_cls = mask_cls.clone()
                loss = None
                for i in range(3):
                    sent_scores = self.model.encoder(sents_vec, memo_cls).squeeze(-1)
                    max_index = torch.argsort(-sent_scores, 1)[:, 0].view(-1, 1)
                    batch_size = src.size()[0]
                    memo_query = sents_vec.gather(1, max_index.expand((batch_size, sents_vec.size()[-1])).unsqueeze(1))
                    sents_vec = self.model.memory(sents_vec, memo_query)
                    loss_1 = self.loss(sent_scores, labels.float())
                    loss_1 = (loss_1*memo_cls.float()).sum()
                    if i == 0:
                        loss = loss_1
                    else:
                        loss += loss_1
                    memo_cls[torch.arange(memo_cls.size()[0]).unsqueeze(1), max_index] = 0
                loss = loss/3
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def memory_test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        print('memory test')
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        self.count+=1
                        if self.count > 50000:
                            break
                        src = batch.src
                        labels = batch.labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask
                        mask_cls = batch.mask_cls
                        topic_label = batch.topic_pro


                        gold = []
                        pred = []

                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            ################################################### topic predict
                            top_vec = self.model.bert(src, segs, mask)
                            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
                            sents_vec = sents_vec * mask_cls[:, :, None].float()
                            pre_label = self.model.topic_predictor(sents_vec).squeeze(-1)
                            # loss
                            # topic_loss = self.loss(pre_label, topic_label.float())
                            # topic_loss = topic_loss.sum()
                            # topic vector

                            topic_vec = pre_label.mm(self.model.topic_embedding).unsqueeze(1)
                            # interactive

                            sents_vec = self.model.memory(sents_vec, topic_vec)

                            sent_scores = self.model.encoder(sents_vec, mask_cls).squeeze(-1)

                            sent_scores = sent_scores + mask_cls.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                        # selected_ids = np.sort(selected_ids,1)
                        # selected_ids: torch.Size([5, 17])
                        # select the top 3
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if (len(batch.src_str[i]) == 0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if (j >= len(batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if (self.args.block_trigram):
                                    if (not _block_tri(candidate, _pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if (self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip() + '\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip() + '\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)
        return stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:

                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                sent_scores, mask, _ = self.model(src, segs, clss, mask, mask_cls)


                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        src = batch.src
                        labels = batch.labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask
                        mask_cls = batch.mask_cls


                        gold = []
                        pred = []

                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            sent_scores, mask, _ = self.model(src, segs, clss, mask, mask_cls)

                            loss = self.loss(sent_scores, labels.float())
                            loss = (loss * mask.float()).sum()
                            batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)

                            sent_scores = sent_scores + mask.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                        # selected_ids = np.sort(selected_ids,1)
                        # selected_ids: torch.Size([5, 17])
                        # select the top 3
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if(len(batch.src_str[i])==0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if(j>=len( batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if(self.args.block_trigram):
                                    if(not _block_tri(candidate,_pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if(self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip()+'\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)
        return stats

    def _save(self, step):
        real_model = self.model
        decoder = self.decoder
        get_s_t = self.get_s_t
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        decoder_state_dict = decoder.state_dict()
        get_s_state_dict = get_s_t.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'decoder': decoder_state_dict,
            'get_s_t': get_s_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _train_save(self, step):
        real_model = self.model
        model_state_dict = real_model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    def oracle(self, test_iter, cal_lead=False, cal_oracle=True, each=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False


        stats = Statistics()
        if each==True:
            _name='each'
        else:
            _name=' '
        gold_path = '%s.gold' % (self.args.result_path)
        oracle_path = '%s.oracle%s' % (self.args.result_path, _name)

        print(gold_path)
        print(oracle_path)
        with open(oracle_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        src = batch.src
                        labels = batch.labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask
                        mask_cls = batch.mask_cls

                        gold = []
                        pred = []

                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            rouge = Rouge()
                            selected_ids= []
                            for i, _ in enumerate(batch.src_str):
                                cur_src = batch.src_str[i]
                                cur_tgt = batch.tgt_str[i]
                                cur_tgt = cur_tgt.split('<q>')
                                # print(cur_tgt)
                                # tgt_num = len(cur_tgt)
                                _selected = []
                                _score =[]
                                # print(len(cur_tgt))
                                for m, tgt_seg in enumerate(cur_tgt):
                                    rouge_score = []
                                    for _,  src_seg in enumerate(cur_src):
                                        # print(tgt_seg)
                                        # print(src_seg)
                                        # print(type(tgt_seg))
                                        # print(type(src_seg))

                                        rouge_score.append(rouge.get_scores(tgt_seg.strip(), src_seg.strip())[0]["rouge-1"]['f'])
                                    cur_index = rouge_score.index(max(rouge_score))

                                    if cur_index not in _selected:
                                        _selected.append(cur_index)
                                        _score.append(max(rouge_score))
                                if len(_selected) != 0:
                                    _score = np.array(_score)
                                    _selected = np.array(_selected)
                                    _selected = list(_selected[_score.argsort()[-5:][::-1]])
                                    # print(_selected)
                            selected_ids.append(_selected)


                        # selected_ids = np.sort(selected_ids,1)
                        # selected_ids: torch.Size([5, 17])
                        # select the top 3
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if(len(batch.src_str[i])==0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if(j>=len( batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if(self.args.block_trigram):
                                    if(not _block_tri(candidate,_pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if(self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip()+'\n')
        if(self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, oracle_path, gold_path)
            logger.info('Rouges at \n%s' % (rouge_results_to_str(rouges)))
        # self._report_step(0, step, valid_stats=stats)
        return stats



    def cluster_test(self, cluster_iter, step):
        can_path = '%s_step%d.candidate'%(self.args.cluster_result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.cluster_result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in cluster_iter:
                        src = batch.src
                        labels = batch.labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask
                        mask_cls = batch.mask_cls

                        gold = []
                        candidate = []

                        # torch.Size([5, 512, 768])
                        top_vec = self.model.bert(src, segs, mask)
                        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
                        # ([5, 21, 768])
                        sents_vec = sents_vec * mask_cls[:, :, None].float()
                        sents_vec = sents_vec.detach().cpu().numpy()
                        # print(np.shape(sents_vec))

                        # pool = Pool()
                        # results=[]
                        # for i, heap in enumerate(sents_vec):
                        #     real_sen_up = len(batch.src_str[i])
                        #     real_heap = heap[:real_sen_up,]
                        #     result = pool.apply_async(get_position, args=(real_heap,))
                        #     results.append(result)
                        # pool.close()
                        # pool.join()
                        #

                        results=[]
                        for i, heap in enumerate(sents_vec):
                            # print(type(heap))
                            # print(heap)
                            # print(np.shape(heap))
                            real_sen_up = len(batch.src_str[i])
                            real_heap = heap[:real_sen_up,]
                            result = get_position(real_heap)
                            results.append(result)


                        for i, result in enumerate(results):
                            cur_index = result
                            # print(cur_index)
                            _candidate = []
                            # _gold = []
                            for j in cur_index:
                                _candidate.append(batch.src_str[i][j].strip())
                            _candidate = '<q>'.join(_candidate)
                            gold.append(batch.tgt_str[i])
                            candidate.append(_candidate)

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(candidate)):
                            save_pred.write(candidate[i].strip()+'\n')

        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            print(rouges)
            print('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        return rouges['rouge_1_f_score']


    def abs_train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):

        logger.info('Start training...')
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        if self.interval:
            for p in self.model.parameters():
                p.requires_grad = False
            self.alter = False

        while step <= train_steps:
            reduce_counter = 0
            if self.interval:
                if step > 100 and not self.alter:
                    for p in self.model.parameters():
                        p.requires_grad = True
                    self.alter = True
            for i, batch in enumerate(train_iter):
                # don't update the extractive model'parameter for the first 5000 step
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    # logger.info('---------------------------step----- %s' % step)
                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed.all_gather_list
                                                (normalization))

                        self._abs_gradient_accumulation(
                        true_batchs, normalization, total_stats,
                        report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    # @torchsnooper.snoop()
    def _abs_gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):

        if self.grad_accum_count > 1:
            self.model.zero_grad()
            self.decoder.zero_grad()
            self.get_s_t.zero_grad()

        for batch in true_batchs:

            if self.grad_accum_count == 1:
                self.model.zero_grad()
                self.decoder.zero_grad()
                self.get_s_t.zero_grad()
            data_a = time()
            src = batch.src

            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls

            tgt_idxs = batch.tgt_idxs
            decoder_tgt_idxs = batch.decoder_tgt_idxs
            logger.debug('src {}'.format(src))
            logger.debug('tgt_idxs {}'.format(tgt_idxs))
            logger.debug('decoder_tgt_idxs {}'.format(decoder_tgt_idxs))


            decoder_step = tgt_idxs.size()[1]
            data_b = time()
            logger.debug('data representation {}'.format(data_b-data_a))
            # extractive
            data_a = time()
            sent_scores, mask_cls, top_vec = self.model(src, segs, clss, mask, mask_cls)
            data_b = time()
            logger.debug('extractive process {}'.format(data_b-data_a))

            data_a = time()
            sent_scores = sent_scores + mask_cls.float()

            # sent_scores = sent_scores.detach().cpu().numpy()
            # sent_scores = sent_scores.detach().cpu().data.numpy()
            selected_ids = torch.argsort(-sent_scores, 1)
            # selected_ids = np.argsort(-sent_scores, 1)

            # mask
            doc_len = src.size(1)
            batch_size = src.size(0)

            clss_fw = torch.cat((clss, torch.full([clss.size(0), 1], doc_len - 1).type(torch.long).to(self.device)),
                                dim=1)
            clss_fw = clss_fw[:, 1:]
            # print('src require grad', src.requires_grad)
            # print('clss_fw require grad', clss_fw.requires_grad)
            # print('selected_ids grad', selected_ids.requires_grad)
            sents_num = min(self.extract_num, selected_ids.size()[1])
            for i in range(sents_num):
                cur_index = selected_ids[:, i]
                for j in range(batch_size):
                    cur_sent = cur_index[j]
                    begin = clss[j, cur_sent] + 1
                    end = clss_fw[j, cur_sent] - 1
                    mask[j, begin:end] = 2
            # torch.full_like()
            # print(mask.type())
            mask_select = torch.where(mask == 2, torch.full_like(mask, 1), torch.full_like(mask, 0)).to(self.device)
            mask = mask_select.type(torch.long)
            # print('mask grad', mask.requires_grad)


            # decoder need
            encoder_outputs = top_vec.to(self.device)
            encoder_feature = top_vec.view(-1, self.model.bert.model.config.hidden_size).to(self.device)

            enc_batch_extend_vocab = src * mask
            enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.device)
            enc_padding_mask = mask.to(self.device)



            # pos_tag
            if self.use_pos_tag:
                pos_tag = torch.full_like(src, 0).to(self.device)
                pos_src_token = src[mask_select]

                pos_src_token = pos_src_token.cpu().data.numpy()
                extracted_word = self.tokenizer.convert_ids_to_tokens(pos_src_token)

                extracted_doc = ' '.join(extracted_word)
                _pos_tag_list = self.nlp.pos_tag(extracted_doc)
                pos_tag_list=[]
                for i in _pos_tag_list:
                    if i[0] != '##' and i[1] in self.pos_dict.keys():
                        pos_tag_list.append(self.pos_dict[i[1]])
                    elif i[0] != '##':
                        pos_tag_list.append(38)
                # pos_tag_list = [self.pos_dict[i[1]] if i[0] != '##' and i[1] in self.pos_dict.keys()
                #                 else 38 for i in pos_tag_list]
                pos_tag_ = torch.LongTensor(pos_tag_list).to(self.device)


            ##################################################################
            # try:
            #     pos_tag[mask_select] = pos_tag_
            # except:
            #     # . . . pos_tag will see an entity
            #     self._error+=1
            #     print(self._error)
            #     continue
            ###############################################################
                # print(self._erroe)
                #
                # print(_pos_tag_list)
                # print(mask_select.sum())
                # print(len(pos_src_token))
                # print(pos_src_token)
                # print(extracted_doc)
                # raise
            # pos_tag[mask_select] = pos_tag_

            dec_padding_mask = torch.where(decoder_tgt_idxs == 0, torch.full_like(decoder_tgt_idxs, 0),
                                           torch.full_like(decoder_tgt_idxs, 1)).to(self.device)
            data_b = time()
            logger.debug('prepare for abstractive{}'.format(data_b-data_a))

            data_a=time()
            # s_t_1     2*hidden_dim
            # 1 x b x h
            s_t_1 = self.get_s_t(encoder_feature, encoder_outputs, mask.type(torch.float))


            # c_t_1
            c_t_1 = torch.zeros((batch_size, self.model.bert.model.config.hidden_size))  # b x 2*hidden
            c_t_1 = c_t_1.to(self.device)

            # print(s_t_1,c_t_1)
            # extra_zeros =None
            # abstractive
            step_losses = []
            _coverage = torch.zeros(src.size()).to(self.device)
            data_b=time()
            logger.debug('get c_t and s_t {}'.format(data_b-data_a))

            first = True
            data_a=time()
            for cur_step in range(decoder_step):
                target = tgt_idxs[:, cur_step]
                y = decoder_tgt_idxs[:, cur_step]
                y = y.unsqueeze(1)  # b x 1

                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y, s_t_1, c_t_1,
                                                                                        encoder_outputs,
                                                                                        encoder_feature,enc_padding_mask,
                                                                                        enc_batch_extend_vocab,
                                                                                        _coverage, cur_step)




                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()

                step_loss = -torch.log(gold_probs + self.eps)  # b x 1

                # coverage
                if True:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, _coverage), 1)
                    step_loss = step_loss + step_coverage_loss
                    _coverage = next_coverage
                step_mask = dec_padding_mask[:, cur_step]
                step_loss = step_loss * step_mask.type(torch.float)
                step_losses.append(step_loss)
            # loss and bp
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / (dec_padding_mask.sum()+1e-12)
            loss = torch.mean(batch_avg_loss)
            # loss.register_hook(lambda g: logger.debug('loss_main grad{}'.format(torch.sum(torch.isnan(g)))))
            # loss.div(float(normalization)).backward()
            data_b=time()
            logger.debug('decoder process {}'.format(data_b-data_a))
            a=time()
            loss.backward()
            b=time()
            logger.debug('bp loss cal {}'.format(b-a))

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        a=time()

        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads=[]
                for _model in [self.model, self.decoder, self.get_s_t]:
                    grads = grads + [p.grad.data for p in _model.parameters() if p.requires_grad and p.grad is not None]
                # print(len(grads))
                # print(grads[1].is_cuda)
                # print(grads[1].device)
                # print(grads[1].size())
                # print(grads[1].type())
                # print(type(grads[1]))
                # print(grads[1])
                # for meta in grads:
                #     print(meta.size())
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            # for _model in [self.model, self.decoder, self.get_s_t]:
            #     grads = grads + [p.grad.data for p in _model.parameters() if p.requires_grad and p.grad is not None]
            self.optim.step()
        b=time()
        logger.debug('bp loss process {}'.format(b - a))


def abs_decode(self, test_iter, step):
    can_path = '%s_step%d.candidate' % (self.args.result_path, step)
    gold_path = '%s_step%d.gold' % (self.args.result_path, step)
    self.beam_decoder = BeamSearch(self.model, self.decoder, self.get_s_t, self.tokenizer, self.nlp,
                                   "cuda", logger=logger)

    with open(can_path, 'w') as save_pred:
        with open(gold_path, 'w') as save_gold:
            with torch.no_grad():
                for batch in test_iter:
                    self.count += 1
                    if self.count > 20:
                        break
                    summary = self.beam_decoder.beam_search(batch)
                    summary = [int(i) for i in summary.tokens[1:]]
                    # logger.info('summary %s' % summary)
                    _pred = self.tokenizer.convert_ids_to_tokens(summary)
                    # logger.info('pred %s' % _pred)
                    _pred = ' '.join(_pred)

                    gold = []
                    pred = []

                    pred.append(_pred)
                    gold.append(batch.tgt_str)

                    # assert len(gold) == len(gold)

                    for i in range(len(gold)):
                        if len(pred[i]) > 0:
                            text = pred[i]
                            # .replace('[PAD]', '').replace('[unused2]', '')
                            text = text.replace('[unused2]', '<q>').replace('[unused1]', '').strip()
                            # new_text = []
                            # text_pool = text.split(' ')
                            # for cur in text_pool:
                            #     if '##' in cur:
                            #         last = new_text[-1]
                            #         last = last + cur.replace('##', '')
                            #         new_text[-1] = last
                            #     else:
                            #         new_text.append(cur)
                            # new_text = ' '.join(new_text)
                            new_text = text.replace(' ##', '')

                            save_gold.write(gold[i][0].strip() + '\n')
                            save_pred.write(new_text.strip() + '\n')
                            # save_pred.write(pred[i].strip()+'\n')

    rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
    logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
