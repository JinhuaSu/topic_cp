import sys



import os
import time

import torch
import numpy as np
import time
import traceback
# use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model, decoder, get_s_t, tokenizer, nlp, device, beam_size=5,
                 max_dec_steps=100, min_dec_steps=10, logger=None):
        self.model = model
        self.decoder = decoder
        self.get_s_t = get_s_t
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.beam_size =beam_size
        self.max_dec_steps = max_dec_steps
        self.start_token = self.tokenizer.convert_tokens_to_ids(['[unused0]'])[0]
        self.end_token = self.tokenizer.convert_tokens_to_ids(['[unused1]'])[0]
        self.min_dec_steps = min_dec_steps
        self.device= device
        self.logger = logger

        self.is_coverage = True



    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):
        #batch should have only one example
        src = batch.src
        labels = batch.labels
        segs = batch.segs
        clss = batch.clss
        mask = batch.mask
        mask_cls = batch.mask_cls
        # self.logger.info(' src%s',src)
        # self.logger.info('segs%s',segs)
        # self.logger.info('clss%s', clss)
        # self.logger.info('mask%s', mask)
        # self.logger.info('mask_cls%s',mask_cls )

        sent_scores, mask_cls, top_vec = self.model(src, segs, clss, mask, mask_cls)
        # self.logger.info('sent_score %s' % sent_scores)
        # self.logger.info('top_Vec %s' % top_vec)

        sent_scores = sent_scores + mask_cls.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)

        # print('sent_scores',sent_scores.shape)

        # mask
        doc_len = src.size(1)
        batch_size = src.size(0)
        clss_fw = torch.cat((clss, torch.full([clss.size(0), 1], doc_len - 1).type(torch.long).to(self.device)), dim=1)
        clss_fw = clss_fw[:, 1:]

        sent_nums = selected_ids.shape[0]
        for i in range(min(5, sent_nums)):
            cur_index = selected_ids[:, i]
            for j in range(batch_size):
                cur_sent = cur_index[j]
                begin = clss[j, cur_sent] + 1
                end = clss_fw[j, cur_sent] - 1
                mask[j, begin:end] = 2
        mask_select = torch.where(mask == 2, torch.full_like(mask, 1), torch.full_like(mask, 0)).to(self.device)
        mask = mask_select.type(torch.long)
        # decoder need
        encoder_outputs = top_vec.to(self.device)
        encoder_feature = top_vec.view(-1, self.model.bert.model.config.hidden_size).to(self.device)
        enc_batch_extend_vocab = src * mask
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.device)
        enc_padding_mask = mask.to(self.device)

        # print(encoder_outputs)
        # print('encoder_outputs', encoder_outputs.size())
        # print('encoder_feature', encoder_feature.size())

        s_t_0 = self.get_s_t(encoder_feature, encoder_outputs, mask.type(torch.float))
        c_t_0 = torch.zeros((batch_size, self.model.bert.model.config.hidden_size)).unsqueeze(0)
        coverage_t_0 = torch.zeros(src.size()).unsqueeze(0)
        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.unsqueeze(0)
        dec_c = dec_c.unsqueeze(0)

        #decoder batch preparation, it has beam_size example initially everything is repeated
        # start_token
        beams = [Beam(tokens=[self.start_token],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if self.is_coverage else None))
                 for _ in range(self.beam_size)]
        results = []
        steps = 0
        try:

            while steps < self.max_dec_steps and len(results) < self.beam_size:
                #
                latest_tokens = [h.latest_token for h in beams]
                # latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                #                  for t in latest_tokens]
                y_t_1 = torch.LongTensor(latest_tokens)
                # self.logger.info('y_t_1 %s' % (y_t_1))
                if self.device == 'cuda':
                    y_t_1 = y_t_1.cuda()
                y_t_1 = y_t_1.unsqueeze(1)
                all_state_h =[]
                all_state_c = []

                all_context = []

                for h in beams:
                    state_h, state_c = h.state
                    all_state_h.append(state_h)
                    all_state_c.append(state_c)

                    all_context.append(h.context)

                s_t_1 = (torch.stack(all_state_h, 0).squeeze().unsqueeze(0), torch.stack(all_state_c, 0).squeeze().unsqueeze(0))
                c_t_1 = torch.stack(all_context, 0)
                #
                c_t_1 = c_t_1.squeeze(1)

                coverage_t_1 = None
                if self.is_coverage:
                    all_coverage = []
                    for h in beams:
                        all_coverage.append(h.coverage)
                    coverage_t_1 = torch.stack(all_coverage, 0)

                b_size = c_t_1.size()[0]
                # print('encoder', encoder_feature.size())
                encoder_outputs = encoder_outputs.index_select(0, torch.zeros(b_size).type(torch.LongTensor).cuda()).cuda()  # b x t x h
                encoder_feature = encoder_outputs.view(-1, self.model.bert.model.config.hidden_size).cuda()
                enc_padding_mask = enc_padding_mask.index_select(0, torch.zeros(b_size).type(torch.LongTensor).cuda()).cuda()
                enc_batch_extend_vocab = enc_batch_extend_vocab.index_select(0, torch.zeros(b_size).type(torch.LongTensor).cuda()).cuda()
                # encoder_outputs
                # torch.Size([1, 102, 768])
                # encoder_feature
                # torch.Size([102, 768])
                # torch.Size([1, 102])
                #
                # print('encoder_outputs', encoder_outputs.size())
                # print('encoder_feature', encoder_feature.size())
                # print('enc_padding_mask', enc_padding_mask.size())
                # print('enc_batch_extend_vocab', enc_batch_extend_vocab.size())
                # print('y_t_1', y_t_1.size())
                # print('s_t_1', s_t_1[0].size())
                # print('c_t_1', c_t_1.size())
                # print('coverage_t_1', coverage_t_1.size())

                # print(y_t_1)
                # print(s_t_1)
                # print(encoder_outputs)
                # 全部都是nan
                final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.decoder(y_t_1, s_t_1, c_t_1,
                                                            encoder_outputs, encoder_feature, enc_padding_mask,
                                                            enc_batch_extend_vocab, coverage_t_1, steps)
                log_probs = torch.log(final_dist)
                # print(log_probs.size())
                # print(log_probs)

                topk_log_probs, topk_ids = torch.topk(log_probs, self.beam_size * 2)
                # self.logger.info('topk_ids %s' % (topk_ids))


                dec_h, dec_c = s_t
                dec_h = dec_h.squeeze()
                dec_c = dec_c.squeeze()

                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(beams)
                # print(topk_ids.size())
                # print(topk_log_probs.size())

                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = (dec_h[i], dec_c[i])
                    context_i = c_t[i]
                    coverage_i = (coverage_t[i] if self.is_coverage else None)

                    for j in range(self.beam_size):  # for each of the top 2*beam_size hyps:
                        try:
                            new_beam = h.extend(token=topk_ids[i, j].item(),
                                       log_prob=topk_log_probs[i, j].item(),
                                       state=state_i,
                                       context=context_i,
                                       coverage=coverage_i)
                        except:
                            # print(topk_ids[i, j].item())
                            self.logger.info('next beam error')
                            continue

                        all_beams.append(new_beam)

                beams = []
                for h in self.sort_beams(all_beams):
                    if h.latest_token == self.end_token:
                        if steps >= self.min_dec_steps:
                            results.append(h)
                    else:
                        beams.append(h)
                    if len(beams) == self.beam_size or len(results) == self.beam_size:
                        break

                steps += 1

                #     msg = traceback.format_exc()
                #     # print(msg)
                #     # self.logger.info('exception %s' % (msg))

            if len(results) == 0:
                results = beams

            beams_sorted = self.sort_beams(results)
            # print(beams_sorted[0])
            # beams_sorted=[[1,1,1]]
        except:
            self.logger.info('one decode process')
            return [0]
        return beams_sorted[0]

