import torch
import json
import random
import itertools
import numpy as np
import ud_features
from pathlib import Path
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, config, training=False, validation=False, testing=False):
        self.config = config
        self.training = training
        # read from configuration
        self.genres = {g: i for i, g in enumerate(self.config['genres'])}
        data_folder = Path(config['data_folder'])
        if training:
            data_file = config['train_data_file']
        elif validation:
            data_file = config['eval_data_file']
        elif testing:
            data_file = config['test_data_file']
        data_path = data_folder.joinpath(data_file)
        # read data from file
        self.data = []
        with open(data_path, encoding='utf8') as file:
            for line in file:
                self.data.append(json.loads(line))
        self.size = len(self.data)

    def get_raw_data(self, item):
        return self.data[item]

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # get requested document
        doc, offset = self.data[item], 0
        # truncate in training if document is too long
        max_segm_num = self.config['max_segm_num']
        if self.training and len(doc['segments']) > max_segm_num:
            doc, offset = self.truncate(doc, max_segm_num)

        # calculate some auxiliary variables
        segms = doc['segments']
        segm_len = torch.tensor([len(s) for s in segms])

        # genre-id
        genre_id = torch.as_tensor([self.genres.get(doc['doc_key'][:2], 0)])

        # speaker-ids
        speakers = list(itertools.chain.from_iterable(doc['speakers']))
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < 20:
                speaker_dict[s] = len(speaker_dict)
        if len(speaker_dict) == 20:
            print(f'Speaker limit reached: {doc["doc_key"]}')
        speaker_ids = torch.as_tensor([speaker_dict.get(s, 3) for s in speakers])

        # read cluster and mentions
        clusters = doc['clusters']
        gold_mentions = sorted(tuple(m) for m in itertools.chain.from_iterable(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}

        # gold-starts / gold-ends
        gold_starts, gold_ends = map(
            torch.as_tensor,
            zip(*gold_mentions) if gold_mentions else ([], [])
        )

        # cluster-ids
        cluster_ids = torch.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                # add one so zero is reserved for 'no cluster' in later matrices
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        # cand-starts / cand-ends
        sent_map = doc['sent_map']
        token_map = doc['token_map']
        morph_map = doc['morph_map']

        if "doc_morph_feats" in doc:
            doc_morph_feats = doc["doc_morph_feats"]
        else:
            doc_morph_feats = self.morph_feats_mod(token_map, morph_map)

        if "cand_starts" in doc and "cand_ends" in doc:
            cand_starts = doc["cand_starts"]
            cand_ends = doc["cand_ends"]
        else:
            cand_starts, cand_ends = self.create_candidates(sent_map, token_map, segm_len)
            self.data[item]["cand_starts"] = cand_starts
            self.data[item]["cand_ends"] = cand_ends

        if "morph_feats" in doc and "morph_feats_mask" in doc:
            morph_feats = doc["morph_feats"]
            morph_feats_mask = doc["morph_feats_mask"]
        else:
            morph_feats, morph_feats_mask = self.morph_feats(cand_starts, cand_ends, token_map, morph_map)
            self.data[item]["morph_feats"] = morph_feats
            self.data[item]["morph_feats_mask"] = morph_feats_mask

        # return all necessary information for training and evaluation
        return segms, segm_len, genre_id, speaker_ids, gold_starts, gold_ends, cluster_ids, cand_starts, cand_ends, morph_feats, morph_feats_mask, doc_morph_feats

    def morph_feats(self, ment_starts, ment_ends, token_map, morph_map):
        ment_starts = ment_starts.tolist()
        ment_ends = ment_ends.tolist()
        feat_vector_size = ud_features.get_ud_features_length()

        morph_feats = []
        morph_feats_mask = []
        for i in range(0, len(ment_starts)):
            men_morph_feats = []
            men_mask = []
            for j in range(ment_starts[i], ment_ends[i] + 1):
                try:
                    sparse_vector = morph_map[str(token_map[j])]
                except:
                    # print("i: " + str(i), flush=True)
                    # print("j: " + str(j), flush=True)
                    # print("token-map: " + str(token_map[j]), flush=True)
                    pass
                feat_vector = torch.zeros(feat_vector_size)
                if len(sparse_vector) > 0:
                    for feat_idx in sparse_vector:
                        feat_vector[feat_idx] = 1
                men_morph_feats.append(feat_vector)
                men_mask.append(1)
            men_morph_feats = torch.stack(men_morph_feats)
            men_mask = torch.IntTensor(men_mask)
            pad_length = self.config['max_ment_width'] - men_morph_feats.size(dim=0)
            men_morph_feats = torch.nn.functional.pad(men_morph_feats, (0, 0, 0, pad_length), "constant", 0)
            men_mask = torch.nn.functional.pad(men_mask, (0, pad_length), "constant", 0)
            morph_feats.append(men_morph_feats)
            morph_feats_mask.append(men_mask)

        morph_feats = torch.stack(morph_feats)
        morph_feats_mask = torch.stack(morph_feats_mask)
        return morph_feats, morph_feats_mask

    def morph_feats_mod(self, token_map, morph_map):
        men_morph_feats = []
        feat_vector_size = ud_features.get_ud_features_length()

        # token_morph_map = [morph_map[str(i)] for i in token_map]

        token_morph_map = []
        for i in token_map:
            try:
                token_morph_map.append(morph_map[str(i)])
            except:
                token_morph_map.append([])

        for token_sparse_vector in token_morph_map:
            feat_vector = torch.zeros(feat_vector_size)
            if len(token_sparse_vector) > 0:
                for feat_idx in token_sparse_vector:
                    feat_vector[feat_idx] = 1
            men_morph_feats.append(feat_vector)
        men_morph_feats = torch.stack(men_morph_feats)

        return men_morph_feats

    def truncate(self, doc, max_segm_num):
        sents = doc['segments']
        segm_num = len(sents)
        sent_len = [len(s) for s in sents]

        # calculated borders for truncation
        sentence_start = random.randint(0, segm_num - max_segm_num)
        sentence_end = sentence_start + max_segm_num
        word_start = sum(sent_len[:sentence_start])
        word_end = sum(sent_len[:sentence_end])

        # update mention indices and remove truncated mentions
        clusters = [[
            (l - word_start, r - word_start)
            for l, r in cluster
            if word_start <= l <= r < word_end
        ] for cluster in doc['clusters']]
        # remove empty clusters
        clusters = [cluster for cluster in clusters if cluster]

        # truncated document with only the necessary information
        trun_doc = {
            'doc_key': doc['doc_key'],
            'segments': sents[sentence_start:sentence_end],
            'sent_map': doc['sent_map'][word_start:word_end],
            'token_map': doc['token_map'][word_start:word_end],
            'speakers': doc['speakers'][sentence_start:sentence_end],
            'clusters': clusters,
            'morph_map': doc['morph_map']
        }

        # return truncated doc and offset
        return trun_doc, sentence_start

    def create_candidates(self, sent_map, token_map, segm_len):
        # calculate sentence lengths
        sent_idx = range(sent_map[-1] + 1)
        sent_len = [len([s for s in sent_map if s == sent]) for sent in sent_idx]

        # calculate all possible mentions
        max_ment_width = self.config['max_ment_width']
        cand_starts, cand_ends = [], []
        offset = 0
        for s in sent_len:
            for i in range(s):
                width = min(max_ment_width, s-i)
                start = i + offset
                cand_starts.extend([start] * width)
                cand_ends.extend(range(start, start+width))
            offset += s

        # candidate boundaries as tensors
        cand_starts = torch.as_tensor(cand_starts)
        cand_ends = torch.as_tensor(cand_ends)

        # set -1 for CLS and SEP token
        token_map[0] = -1
        token_map[-1] = -1
        for sl in np.cumsum(segm_len[:-1]):
            token_map[sl-1:sl+1] = [-1, -1]
        tkn_map = torch.tensor(token_map)

        # create tensor with possible starts
        tkn_map_sh = torch.tensor([-1] + token_map[:-1])
        start_ = tkn_map != tkn_map_sh
        start_ &= tkn_map >= 0
        # create tensor with possible ends
        tkn_map_sh = torch.tensor(token_map[1:] + [-1])
        end_ = tkn_map != tkn_map_sh
        end_ &= tkn_map >= 0

        # apply subtoken rules
        filter = start_[cand_starts] & end_[cand_ends]
        cand_starts = cand_starts[filter]
        cand_ends = cand_ends[filter]

        return cand_starts, cand_ends


class DataLoader(data.DataLoader):

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=DataLoader.collate, batch_size=1, pin_memory=True, **kwargs)

    @staticmethod
    def collate(batch):
        return batch[0]
