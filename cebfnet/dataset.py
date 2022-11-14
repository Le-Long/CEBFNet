import os

import torch
from torch.utils import data as torch_data

from torchdrug import data, datasets, utils
from torchdrug.core import Registry as R

from transformers import AutoTokenizer

from models.utils import InputExample, InputFeatures
from models.utils import truncate_seq_pair, load_intent_datasets, load_intent_examples, sample


class IntentLinkPrediction(data.KnowledgeGraphDataset):
    ENTAILMENT = 'entailment'
    NON_ENTAILMENT = 'non_entailment'
    BELONG_TO = 'belong_to'
    label_list = [ENTAILMENT, NON_ENTAILMENT, BELONG_TO]
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def convert_examples_to_features(self, examples):
        label_map = {label: i for i, label in enumerate(self.label_list)}
        is_roberta = True

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)
            tokens_b = self.tokenizer.tokenize(example.text_b)

            if is_roberta:
                truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 4)
            else:
                truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)

            tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            if is_roberta:
                tokens_b = [self.tokenizer.sep_token] + tokens_b + [self.tokenizer.sep_token]
                segment_ids += [0] * len(tokens_b)
            else:
                tokens_b = tokens_b + [self.tokenizer.sep_token]
                segment_ids += [1] * len(tokens_b)
            tokens += tokens_b

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            features.append([input_ids, input_mask, segment_ids])

        return features

    def load_file(self, train_files, test_files, ood_files):
        inv_train_entity_vocab = {}
        inv_relation_vocab = {self.NON_ENTAILMENT: 0, self.ENTAILMENT: 1, self.BELONG_TO: 2}
        triplets = []
        num_samples = []

        N = self.few_shot_num
        T = self.num_trials

        train_examples, dev_examples = load_intent_datasets(train_files, test_files, True)
        sampled_tasks = [sample(N, train_examples) for i in range(T)]
        oos_dev_examples = load_intent_examples(ood_files, True)

        nli_train_examples = []
        nli_dev_examples = []
        encoder_features = []

        for i in range(T):

            tasks = sampled_tasks[i]
            all_examples = []
            dev_entailment_examples = []
            test_entailment_examples = []

            for e in dev_examples:
                for task in tasks:
                    examples = task['examples']
                    for j in range(len(examples)):
                        if e.label == task["task"]:
                            dev_entailment_examples.append(InputExample(e.text, examples[j], self.ENTAILMENT))
                        else:
                            dev_entailment_examples.append(InputExample(e.text, examples[j], self.NON_ENTAILMENT))
            for oos in oos_dev_examples:
                for task in tasks:
                    examples = task['examples']
                    for j in range(len(examples)):
                        test_entailment_examples.append(InputExample(oos.text, examples[j], self.NON_ENTAILMENT))

            # train_examples
            for task_1 in tasks:
                examples_1 = task_1['examples']
                for j in range(len(examples_1)):
                    same_head_examples = []
                    for task_2 in tasks:
                        examples_2 = task_2['examples']
                        for k in range(len(examples_2)):
                            if task_1 == task_2:
                                same_head_examples.append(InputExample(examples_1[j], examples_2[k], self.ENTAILMENT))
                            else:
                                same_head_examples.append(
                                    InputExample(examples_1[j], examples_2[k], self.NON_ENTAILMENT))
                    assert len(same_head_examples) == N * 10
                    all_examples.extend(same_head_examples)

            nli_train_examples.append(all_examples)
            nli_dev_examples.append(dev_entailment_examples + test_entailment_examples)

        for j in range(T):
            num_sample = 0
            for e in nli_train_examples[j]:
                h_token = e.text_a
                t_token = e.text_b
                r_token = e.label
                if h_token not in inv_train_entity_vocab:
                    inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                h = inv_train_entity_vocab[h_token]
                r = inv_relation_vocab[r_token]
                if t_token not in inv_train_entity_vocab:
                    inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                t = inv_train_entity_vocab[t_token]
                triplets.append((h, t, r))
                num_sample += 1
            num_samples.append(num_sample)
            encoder_features.extend(self.convert_examples_to_features(nli_train_examples[j]))

        inv_test_entity_vocab = inv_train_entity_vocab.copy()
        oos_triplets = []

        for j in range(T):
            num_sample = 0
            num_oos = 0
            for i, e in enumerate(nli_dev_examples[j]):
                h_token = e.text_a
                t_token = e.text_b
                r_token = e.label
                if h_token not in inv_test_entity_vocab:
                    inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                h = inv_test_entity_vocab[h_token]
                r = inv_relation_vocab[r_token]
                if t_token not in inv_test_entity_vocab:
                    inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                t = inv_test_entity_vocab[t_token]
                if i < len(dev_entailment_examples):
                    num_sample += 1
                    triplets.append((h, t, r))
                else:
                    num_oos += 1
                    oos_triplets.append((h, t, r))

            num_samples.append(num_sample)
            encoder_features.extend(self.convert_examples_to_features(nli_dev_examples[j]))

        triplets.extend(oos_triplets)
        # word_net = datasets.WN18RR("../wordnet/")
        # word_triplets = []
        # for word_triplet in word_net[:10000]:
        #     flag = False
        #     head = word_net.entity_vocab[word_triplet[0]]
        #     tail = word_net.entity_vocab[word_triplet[1]]
        #     rela = word_net.relation_vocab[word_triplet[2]]
        #     for entity in inv_test_entity_vocab:
        #         if head in entity:
        #             h_index = inv_test_entity_vocab[entity]
        #             if head not in inv_test_entity_vocab:
        #                 inv_test_entity_vocab[head] = len(inv_test_entity_vocab)
        #             h = inv_test_entity_vocab[head]
        #             if tail not in inv_test_entity_vocab:
        #                 inv_test_entity_vocab[tail] = len(inv_test_entity_vocab)
        #             t = inv_test_entity_vocab[tail]
        #             if rela not in inv_relation_vocab:
        #                 inv_relation_vocab[rela] = len(inv_relation_vocab)
        #             r = inv_relation_vocab[rela]
        #
        #             if (h, h_index, inv_relation_vocab[self.BELONG_TO]) not in triplets:
        #                 word_triplets.append((h, h_index, inv_relation_vocab[self.BELONG_TO]))
        #                 flag = True
        #
        #         if tail in entity:
        #             t_index = inv_test_entity_vocab[entity]
        #             if tail not in inv_test_entity_vocab:
        #                 inv_test_entity_vocab[tail] = len(inv_test_entity_vocab)
        #             t = inv_test_entity_vocab[tail]
        #             if head not in inv_test_entity_vocab:
        #                 inv_test_entity_vocab[head] = len(inv_test_entity_vocab)
        #             h = inv_test_entity_vocab[head]
        #             if rela not in inv_relation_vocab:
        #                 inv_relation_vocab[rela] = len(inv_relation_vocab)
        #             r = inv_relation_vocab[rela]
        #
        #             if (t, t_index, inv_relation_vocab[self.BELONG_TO]) not in triplets:
        #                 word_triplets.append((t, t_index, inv_relation_vocab[self.BELONG_TO]))
        #                 flag = True
        #
        #     if flag:
        #         word_triplets.append((h, t, r))

        train_entity_vocab, inv_train_entity_vocab = self._standarize_vocab(None, inv_train_entity_vocab)
        test_entity_vocab, inv_test_entity_vocab = self._standarize_vocab(None, inv_test_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        self.train_graph = data.Graph(triplets[:num_samples[0]],
                                      num_node=len(test_entity_vocab), num_relation=len(relation_vocab))
        self.valid_graph = data.Graph(triplets[:num_samples[0]],
                                      num_node=len(test_entity_vocab), num_relation=len(relation_vocab))
        self.test_graph = self.valid_graph
        self.graph = data.Graph(triplets[:], num_node=len(test_entity_vocab), num_relation=len(relation_vocab))
        assert self.graph.num_edge == num_samples[0] * T + num_samples[T] * T + num_oos * T

        with self.valid_graph.graph():
            self.valid_graph.query = torch.zeros(1, 25, 768)
        with self.valid_graph.node():
            self.valid_graph.boundary = torch.zeros(len(test_entity_vocab), 25, 768)

        self.num_samples = [(len(train_entity_vocab) * self.batch),
                            ((len(test_entity_vocab) - len(train_entity_vocab)) * self.batch), 0]
        self.sampled_tasks = sampled_tasks
        self.dev_examples = dev_examples
        self.oos_dev_examples = oos_dev_examples
        self.train_entity_vocab = train_entity_vocab
        self.test_entity_vocab = test_entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_train_entity_vocab = inv_train_entity_vocab
        self.inv_test_entity_vocab = inv_test_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab
        self.encoder_features = encoder_features
        self.triplets = self.graph.edge_list

    def __getitem__(self, index):
        features = []
        tail = []
        rela = []
        flag = 0
        for i, triplet in enumerate(self.triplets):
            if triplet[0] == int(index / self.batch):
                flag += 1
                offset = index % self.batch
                if (flag > offset * len(self.train_entity_vocab) / self.batch) and \
                        (flag <= (offset + 1) * len(self.train_entity_vocab) / self.batch):
                    features.append(self.encoder_features[i])
                    tail.append(triplet[1])
                    rela.append(triplet[2])
        return tuple((int(index / self.batch), torch.as_tensor(tail), torch.as_tensor(rela), torch.as_tensor(features)))

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = []
            for i in range(self.num_trials):
                trial = torch_data.Subset(self, range(offset, offset + num_sample))
                split.append(trial)
                offset += num_sample
            splits.append(split)
        return splits


@R.register("datasets.CLINCbankingLinkPrediction")
class CLINCbankingLinkPrediction(IntentLinkPrediction):
    train_path = "train"
    test_path = "test"
    ood_path = "id-oos/test"
    max_seq_length = 128
    batch = 2

    def __init__(self, path, few_shot_num, num_trials, verbose=1):
        self.few_shot_num = few_shot_num
        self.num_trials = num_trials
        self.transform = None
        train_path = os.path.join(path, self.train_path)
        test_path = os.path.join(path, self.test_path)
        ood_path = os.path.join(path, self.ood_path)
        self.load_file(train_path, test_path, ood_path)
