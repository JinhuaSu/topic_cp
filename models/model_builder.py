
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerInterEncoder, Classifier, RNNEncoder
from models.optimizers import Optimizer
import numpy as np
import torchsnooper
import torch.nn.functional as F
from stanfordcorenlp import StanfordCoreNLP
# from pytorch_pretrained_bert import BertTokenizer
from models.tokenization import BertTokenizer
import copy
from torch.nn.init import xavier_uniform_
# from others.logging import logger
from gensim import corpora, models
import re


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from or args.recover_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)
    if isinstance(model, list):
        tmp = []
        for _model in model:
            tmp.extend(list(_model.named_parameters()))
        optim.set_parameters(tmp)
    else:
        optim.set_parameters(list(model.named_parameters()))

    if args.train_from or args.recover_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


def get_pos_dic():
    POS_TAG = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
               'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '[CLS]', '[SEP]']
    pos_dict = dict(zip(POS_TAG, list(range(len(POS_TAG)))))
    return pos_dict


class pair_loss(nn.Module):
    def __init__(self):
        super(pair_loss, self).__init__()

    def forward(self, sent_score, label, mask_cls, max_index):
        max_num = sent_score[torch.arange(sent_score.size()[0]).view(-1,1), max_index]
        refer = max_num.expand_as(sent_score)
        # loss_matrix = sent_score - refer
        # loss_matrix = torch.where(loss_matrix>0, loss_matrix, torch.full_like(loss_matrix,0))
        loss_matrix = refer - sent_score
        loss_matrix = torch.log1p(torch.exp(-loss_matrix))
        # loss_matrix = torch.where(loss_matrix > 0, loss_matrix, torch.full_like(loss_matrix,0))
        loss_final = loss_matrix * mask_cls.float()
        loss_final = loss_final * (1 - label)
        return loss_final.sum()



class get_initial_s(nn.Module):
    def __init__(self, emb_dim, device=None):
        super(get_initial_s, self).__init__()
        self.emb_dim = emb_dim
        self.v = nn.Linear(emb_dim, 1, bias=False)
        self.reduce = nn.Linear(emb_dim, emb_dim // 2)
        self.eps = 1e-12

        if device != None:
            self.to(device)

    def forward(self, encoder_features, encoder_outputs, mask_select):
        t_k = encoder_outputs.size()[1]
        e = F.tanh(encoder_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        attn_dist_ = F.softmax(scores, dim=1) * mask_select  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / (normalization_factor + self.eps)
        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, self.emb_dim)  # B x 2*hidden_dim
        s_t_1 = F.relu(self.reduce(c_t))  # B x hidden_dim
        # 1 x b x h
        s_t_1 = (s_t_1.unsqueeze(0), s_t_1.unsqueeze(0))
        return s_t_1

    def load_cp(self, pt):
        self.load_state_dict(pt['get_s_t'], strict=True)


class Attention(nn.Module):
    def __init__(self, hidden_dim, is_coverage,):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.is_coverage = is_coverage
        # attention
        if self.is_coverage:
            self.W_c = nn.Linear(1, hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.eps = 1e-12

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, mask_select, coverage):
        b, t_k, n = list(encoder_outputs.size())
        # encoder_ouputs b x t x h
        # encoder_feature b*t x h
        # s_t_hat b x 2*h
        # s_t_hat torch.Size([3, 768])

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim


        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim


        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim


        if self.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k


        attn_dist_ = F.softmax(scores, dim=1) * mask_select.type(torch.float)  # B x t_k

        normalization_factor = attn_dist_.sum(1, keepdim=True)

        attn_dist = attn_dist_ / (normalization_factor + self.eps)

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 2 x hidden_dim
        c_t = c_t.view(-1, self.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    # assert hidden_dim is half of emb_dim
    def __init__(self, hidden_dim, vocab_size, emb_dim, embedding, device, pointer_gen=True):
        super(Decoder, self).__init__()
        self.attention_network = Attention(hidden_dim, is_coverage=True)
        self.pointer_gen = pointer_gen
        # decoder
        # init_wt_normal(self.embedding.weight)
        self.hidden_dim = hidden_dim

        self.x_context = nn.Linear(self.hidden_dim * 2 + emb_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        # init_lstm_wt(self.lstm)

        if pointer_gen:
            self.p_gen_linear = nn.Linear(self.hidden_dim * 4 + emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.out2 = nn.Linear(self.hidden_dim, vocab_size)
        #

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        self.embedding = embedding
        if device != None:
            self.to(device)

    def forward(self, y_t_1, s_t_1, c_t_1, encoder_outputs, encoder_feature, mask_select,
                enc_batch_extend_vocab, coverage, cur_step=0):

        # print(y_t_1.size())  # b x 1
        # print(s_t_1[0].size())  # (1 x b x h/2,
        # print(c_t_1.size())  # b x h
        # print(encoder_feature.size())  # b*t x h
        # print(encoder_outputs.size())  # b x t x h
        # print(mask_select.size())  # b x t
        # print(enc_batch_extend_vocab.size())  # b x t
        # print(coverage.size())  # b x 1 x t

        if not self.training and cur_step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                                 c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              mask_select, coverage)
            coverage = coverage_next
        # print(self.embedding.size())

        y_t_1_embd = self.embedding(y_t_1)
        y_t_1_embd = y_t_1_embd.squeeze(1)


        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        # s_t_1 is the encoding vector of input at the begin of decoding


        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)


        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                             c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               mask_select, coverage)
        if self.training or cur_step > 0:
            coverage = coverage_next

        p_gen = None
        if self.pointer_gen:

            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)

            p_gen = self.p_gen_linear(p_gen_input)
            # logger.warning('p_gen_input  sum {} ,max number {}'.format(torch.sum(p_gen_input), torch.max(p_gen_input)))
            # logger.warning('p_gen_linear weight sum {}; maximum number{}'.format(torch.sum(self.p_gen_linear.weight),
            #                                                                      torch.max(self.p_gen_linear.weight)))
            #
            # self.p_gen_linear.weight.register_hook(lambda g: logger.warning('p_gen_linear weight grad have {} nan value, sum {}, max {}'
            #                                              .format(torch.sum(torch.isnan(g)), torch.sum(g), torch.max(g))))
            # p_gen.register_hook(lambda g: logger.warning('p_gen grad have {} nan value, sum {}, max {}'
            #                                              .format(torch.sum(torch.isnan(g)), torch.sum(g), torch.max(g))))
            # # np.save('/home1/bqw/p_gen.npy', p_gen.cpu().data.numpy())
            # # np.save('/home1/bqw/p_gen_input.npy', p_gen_input.cpu().data.numpy())
            # # np.save('/home1/bqw/weight.npy', list(self.p_gen_linear.parameters())[0].cpu().data.numpy())
            p_gen = F.sigmoid(p_gen)
            # logger.debug('--p_gen-------{}'.format(p_gen))

        output = torch.cat((lstm_out.view(-1, self.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

    def load_cp(self, pt):
        self.load_state_dict(pt['decoder'], strict=True)


class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            # self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            self.model = BertModel.from_pretrained(temp_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask =mask)
        top_vec = encoded_layers[-1]
        return top_vec


class Memory(nn.Module):
    def __init__(self, device, hop, d_model):
        super(Memory, self).__init__()
        self.hops = hop
        vt = nn.Linear(2*d_model, d_model)
        self.vt_layers = clones(vt, hop)
        Wq = nn.Linear(d_model, d_model)
        self.Wq_layers = clones(Wq, hop)
        Ws = nn.Linear(d_model, d_model)
        self.Ws_layers = clones(Ws, hop)

        self.to(device)

    def forward(self, src, query):
        batch_size, t, _ = src.size()
        for i in range(self.hops):
            src_proj = self.Ws_layers[i](src)
            query_proj = self.Wq_layers[i](query)
            query_proj = query_proj.expand(src.size())

            cat_proj = torch.cat((src_proj, query_proj), -1)
            src = self.vt_layers[i](torch.tanh(cat_proj))
        return src


class Key_memory(nn.Module):
    def __init__(self, device, hop, d_model, drop_out):
        super(Key_memory, self).__init__()
        self.hops = hop
        self.softmax = nn.Softmax(dim=2)

        dropout = nn.Dropout(drop_out)
        self.dropout_layers = clones(dropout, hop)
        Wp = nn.Linear(d_model, d_model)
        self.Wp_layers = clones(Wp, hop)

        self.to(device)

    def forward(self, src, query):
        # src b x n x h
        # query w x h
        # b x n x w
        memory_mat = src
        for i in range(self.hops):
            project_mat = torch.matmul(memory_mat, query.transpose(0, 1))
            project_mat = self.softmax(project_mat)
            project_mat = self.dropout_layers[i](project_mat)
            # b x n x h
            memory_mat = torch.matmul(project_mat, query)
            memory_mat = self.Wp_layers[i](memory_mat + src)

        return memory_mat


class Doc_extractor(nn.Module):
    def __init__(self, hidden_dim, dropout, num_layers, device):
        super(Doc_extractor, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, dropout=dropout, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.project = nn.Linear(2*hidden_dim, hidden_dim)
        self.to(device)
        
    def forward(self, sents_vec, mask_cls):
        doc_vec = torch.mean(sents_vec, 1)
        # sents_vec, hidden = self.lstm(sents_vec)
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        # doc_vec = sents_vec.sum(1)
        # doc_vec = self.project(doc_vec)
        return doc_vec


class Topic_predictor(nn.Module):
    def __init__(self, args, d_model, device, topic_num):
        super(Topic_predictor, self).__init__()
        self.project = nn.Linear(d_model, topic_num)
        self.to(device)

    def forward(self, doc_vec):
        label = self.project(doc_vec).squeeze(-1)
        label = F.softmax(label, dim=1)
        return label

    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None, topic_num=10):
        super(Summarizer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                                  never_split=(
                                                  '[SEP]', '[CLS]', '[PAD]', '[unused0]', '[unused1]', '[unused2]',
                                                  '[UNK]'),
                                                  no_word_piece=True)
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        self.memory = Memory(device, 1, self.bert.model.config.hidden_size)
        self.key_memory = Key_memory(device, 1, self.bert.model.config.hidden_size, args.dropout)
        self.doc_extractor = Doc_extractor(self.bert.model.config.hidden_size, args.dropout, 1, device)
        self.topic_predictor = Topic_predictor(args, self.bert.model.config.hidden_size, device, topic_num)

        # self.topic_embedding = nn.Embedding(topic_num, self.bert.model.config.hidden_size)
        # todo transform to normal weight not embedding
        
        self.topic_embedding, self.topic_word, self.topic_word_emb = self.get_embedding(self.bert.model.embeddings)
        self.topic_embedding.requires_grad = True
        self.topic_word_emb.requires_grad = True
        self.topic_embedding = self.topic_embedding.to(device)
        self.topic_word_emb = self.topic_word_emb.to(device)
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif (args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif (args.encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.bert.model = BertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        top_vec = self.bert(x, segs, mask)

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)

        return sent_scores, mask_cls, top_vec
    # todo
    def get_embedding(self, embedding):
        lda_model_tfidf = models.ldamodel.LdaModel.load('/home1/bqw/sum/sum_topic_10.model')
        topic = lda_model_tfidf.show_topics()
        embed_list = []
        topic_word = []
        topic_word_emb = []
        back = re.compile(r'["](.*?)["]')
        for meta_topic in topic:
            tmp_embedding = None
            text = meta_topic[1]
            word_list = re.findall(back, text)
            count = 0
            for i, word in enumerate(word_list):
                if word not in self.tokenizer.vocab:
                    continue
                topic_word.append(word)
                if tmp_embedding is None:
                    tmp_embedding = embedding(torch.LongTensor([[self.tokenizer.vocab[word]]])).squeeze().detach().numpy()
                    topic_word_emb.append(tmp_embedding)
                    count += 1
                else:
                    tmp_embedding += embedding(torch.LongTensor([[self.tokenizer.vocab[word]]])).squeeze().detach().numpy()
                    topic_word_emb.append(tmp_embedding)
                    count += 1
            tmp_embedding = tmp_embedding / count
            embed_list.append(tmp_embedding)
        return torch.FloatTensor(embed_list), topic_word, torch.FloatTensor(topic_word_emb)

    def sent_kmax(self, sents_vec, clss, clss_fw, device, doc_len):
        sent = []
        # print('sents_vec',sents_vec.size())
        for vec, cl_num, cl_fw in zip(sents_vec, clss, clss_fw):
            # print(vec.size())
            if cl_fw <= 0 or cl_fw == doc_len:
                sent.append(torch.zeros(vec.size(1)).unsqueeze(0).to(device))
            else:
                sent.append(vec.narrow(0, cl_num, cl_fw))
        # sent = [vec.narrow(cl_num, cl_fw) for vec, cl_num, cl_fw in zip(temp, clss, clss_fw)]
        # for i in sent:
        #     print(i.size())
        hidden_num = sents_vec.size(-1)

        # print(hidden_num)
        # print(sent[0].unsqueeze(0).unsqueeze(0).size())
        func_list = nn.ModuleList([nn.Conv2d(1, 1, (3, 9), padding=(2, 4))])
        sent = [F.relu(conv(x.unsqueeze(0).unsqueeze(0))) for conv in func_list for x in sent]

        func_list = nn.ModuleList([nn.AvgPool2d(1, )])
        # print(sent[0].size())
        sent = [torch.mean(x.squeeze(0).squeeze(0).type(torch.float), dim=0).unsqueeze(0) for x in sent]
        # print(sent[0].size())
        # func_list = nn.ModuleList([nn.Conv2d(1, 1, (3, 5), padding=(2, 2))])
        # sent = [conv(x.unsqueeze(0).unsqueeze(0)) for conv in func_list for x in sent]
        #
        #
        # func_list = nn.ModuleList([kmax_pooling(0, 1)])
        # sent = [k_pool(x.squeeze(0).squeeze(0)) for k_pool in func_list for x in sent]
        # func_list = nn.ModuleList([kmax_pooling(0, 1)])
        # sent_pool = [k_pool(x) for k_pool in func_list for x in sent]

        # print(sent_pool[0].size())
        sent_pool = torch.cat(sent, dim=0).unsqueeze(1)
        # print('sent_pool:', sent_pool.size())
        return sent_pool


# @torchsnooper.snoop()
def sent_kmax(sents_vec, clss, clss_fw, device, doc_len):
    sent = []
    # print('sents_vec',sents_vec.size())
    for vec, cl_num, cl_fw in zip(sents_vec, clss, clss_fw):
        # print(vec.size())
        if cl_fw <= 0 or cl_fw == doc_len:
            sent.append(torch.zeros(vec.size(1)).unsqueeze(0).to(device))
        else:
            sent.append(vec.narrow(0, cl_num, cl_fw))
    # sent = [vec.narrow(cl_num, cl_fw) for vec, cl_num, cl_fw in zip(temp, clss, clss_fw)]
    # for i in sent:
    #     print(i.size())
    hidden_num = sents_vec.size(-1)

    # print(hidden_num)
    # print(sent[0].unsqueeze(0).unsqueeze(0).size())
    func_list = nn.ModuleList([nn.Conv2d(1, 1, (3, 9), padding=(2, 4))])
    sent = [F.relu(conv(x.unsqueeze(0).unsqueeze(0))) for conv in func_list for x in sent]


    func_list = nn.ModuleList([nn.AvgPool2d(1, )])
    # print(sent[0].size())
    sent = [torch.mean(x.squeeze(0).squeeze(0).type(torch.float), dim=0).unsqueeze(0) for x in sent]
    # print(sent[0].size())
    # func_list = nn.ModuleList([nn.Conv2d(1, 1, (3, 5), padding=(2, 2))])
    # sent = [conv(x.unsqueeze(0).unsqueeze(0)) for conv in func_list for x in sent]
    #
    #
    # func_list = nn.ModuleList([kmax_pooling(0, 1)])
    # sent = [k_pool(x.squeeze(0).squeeze(0)) for k_pool in func_list for x in sent]
    # func_list = nn.ModuleList([kmax_pooling(0, 1)])
    # sent_pool = [k_pool(x) for k_pool in func_list for x in sent]


    # print(sent_pool[0].size())
    sent_pool = torch.cat(sent,dim=0).unsqueeze(1)
    # print('sent_pool:', sent_pool.size())
    return sent_pool


class kmax_pooling(nn.Module):
    def __init__(self, dim, k=1):
        super(kmax_pooling, self).__init__()
        self.dim = dim
        self.k = k

    def forward(self, x):
        index = x.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
        return x.gather(self.dim, index)
