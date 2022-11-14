import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from tqdm import tqdm

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from models.utils import calc_oos_f1, InputExample
from models.utils import THRESHOLDS, print_results
from models.utils import calc_oos_precision, calc_in_acc, calc_oos_recall, calc_oos_f1

@R.register("tasks.KnnPrediction")
class KnnPrediction(tasks.Task, core.Configurable):
    _option_members = ["criterion", "metric"]

    def __init__(self, model, criterion="bce", metric=("auroc", "ap"), num_negative=128, strict_negative=True):
        super(KnnPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.num_negative = num_negative
        self.strict_negative = strict_negative

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_node = dataset.graph.num_node
        train_mask = train_set.indices
        valid_mask = valid_set.indices
        train_graph = dataset.train_graph
        valid_graph = dataset.valid_graph
        test_graph = dataset.test_graph
        all_graph = dataset.graph
        self.register_buffer("train_graph", train_graph.undirected())
        self.register_buffer("valid_graph", valid_graph.undirected())
        self.register_buffer("test_graph", test_graph.undirected())
        # self.register_buffer("graph", all_graph.undirected())

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred.squeeze(), target.squeeze(), reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            loss = loss.mean()
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss

        return all_loss, metric

    def predict_and_target(self, batch, all_loss=None, metric=None):
        pos_h_index, t_index, r_index, features = batch
        batch_size = t_index.shape[1]
        assert not torch.any(torch.isnan(pos_h_index))
        assert not torch.any(torch.isnan(t_index))
        assert not torch.any(torch.isnan(r_index))

        h_index = pos_h_index.unsqueeze(-1)

        graph = getattr(self, "%s_graph" % self.split)

        pred = self.model(graph, h_index, t_index, r_index, features, all_loss=all_loss, metric=metric)
        target = r_index.float()
        assert pred.shape == target.shape
        return pred, target

    def evaluate(self, dataset, j, thresholds=THRESHOLDS):
        self.model.eval()
        trained = dataset.sampled_tasks[j]
        vocab = dataset.inv_test_entity_vocab
        val = dataset.dev_examples[:30]
        oos = dataset.oos_dev_examples[:20]
        test = oos + val

        in_domain_preds = []
        oos_preds = []
        for i, test_e in enumerate(tqdm(test, desc='Intent examples')):
            nli_input = []
            h_index = [vocab[test_e.text], ]
            t_index = []
            for t in trained:
                for e in t['examples']:
                    nli_input.append(InputExample(test_e.text, e))
                    t_index.append(vocab[e])
            features = torch.tensor([dataset.convert_examples_to_features(nli_input)], device=self.device)
            h_index = torch.tensor([h_index], device=self.device)
            t_index = torch.tensor([t_index], device=self.device)

            graph = self.valid_graph
            pred = []
            end = 25
            for j in range(0, 2):
                pred.append(self.model(graph, h_index, t_index[:, end*j:end*(j+1)], features=features[:, end*j:end*(j+1)]))
            maxScore, maxIndex = torch.cat(pred, dim=1).squeeze().max(dim=0)

            maxScore = torch.sigmoid(maxScore).item()
            maxIndex = maxIndex.item()

            index = -1
            for t in trained:
                for e in t['examples']:
                    index += 1
                    if index == maxIndex:
                        intent = t['task']
            if i < len(oos):
                oos_preds.append((maxScore, intent))
            else:
                in_domain_preds.append((maxScore, intent))

        print(in_domain_preds)
        print(oos_preds)
        in_acc = calc_in_acc(val, in_domain_preds, THRESHOLDS)
        oos_recall = calc_oos_recall(oos_preds, THRESHOLDS)
        oos_prec = calc_oos_precision(in_domain_preds, oos_preds, THRESHOLDS)
        oos_f1 = calc_oos_f1(oos_recall, oos_prec)

        print_results(THRESHOLDS, in_acc, oos_recall, oos_prec, oos_f1)
        return in_acc[4]

    def predict(self, dataset, j):
        self.model.eval()
        trained = dataset.sampled_tasks[j]
        vocab = dataset.inv_test_entity_vocab
        val = dataset.dev_examples
        oos = dataset.oos_dev_examples
        test = oos + val

        in_domain_preds = []
        oos_preds = []
        for i, test_e in enumerate(tqdm(test, desc='Intent examples')):
            nli_input = []
            h_index = [vocab[test_e.text], ]
            t_index = []
            for t in trained:
                for e in t['examples']:
                    nli_input.append(InputExample(test_e.text, e))
                    t_index.append(vocab[e])
            features = torch.tensor([dataset.convert_examples_to_features(nli_input)], device=self.device)
            h_index = torch.tensor([h_index], device=self.device)
            t_index = torch.tensor([t_index], device=self.device)

            graph = self.valid_graph
            pred = []
            end = 25
            for j in range(0, 2):
                pred.append(self.model(graph, h_index, t_index[:, end*j:end*(j+1)], features=features[:, end*j:end*(j+1)]))
            maxScore, maxIndex = torch.cat(pred, dim=1).squeeze().max(dim=0)

            maxScore = torch.sigmoid(maxScore).item()
            maxIndex = maxIndex.item()

            index = -1
            for t in trained:
                for e in t['examples']:
                    index += 1
                    if index == maxIndex:
                        intent = t['task']
            if i < len(oos):
                oos_preds.append((maxScore, intent))
            else:
                in_domain_preds.append((maxScore, intent))

        print(in_domain_preds)
        print(oos_preds)
        in_acc = calc_in_acc(val, in_domain_preds, THRESHOLDS)
        oos_recall = calc_oos_recall(oos_preds, THRESHOLDS)
        oos_prec = calc_oos_precision(in_domain_preds, oos_preds, THRESHOLDS)
        oos_f1 = calc_oos_f1(oos_recall, oos_prec)

        print_results(THRESHOLDS, in_acc, oos_recall, oos_prec, oos_f1)
        return in_acc[4]
