import torch.nn.functional as F
import medmnist

class MyEvaluator:

    num_class = {
        "pathmnist": 9,
        "chestmnist": 14,
        "dermamnist": 7,
        "octmnist": 4,
        "pneumoniamnist": 2,
        "retinamnist": 5,
        "breastmnist": 2,
        "bloodmnist": 8,
        "tissuemnist": 8,
        "organamnist": 11,
        "organcmnist": 11,
        "organsmnist": 11,
    }

    overlap_sets = ["organamnist", "organcmnist", "organsmnist"]

    def __init__(self, data_flag, subset="test"):
        self.data_flag = data_flag
        self.subset = subset
        self.evaluators = self.load_all_evaluators()
        self.mevaluator = medmnist.Evaluator(data_flag, subset)

    def get_evaluator(self, data_flag):
        return medmnist.Evaluator(data_flag, self.subset)
    
    def load_all_evaluators(self):
        evaluators = {}
        for data_flag in self.num_class.keys():
            evaluators[data_flag] = self.get_evaluator(data_flag)
        return evaluators
    
    def evaluate(self, predictions, subset_id):
        assert subset_id.shape[0] == predictions.shape[0]

        # split predictions into subsets according to self.num_clss
        y_scores = {}
        base_cls = 0
        for i, (data_flag, ncls) in enumerate(self.num_class.items()):
            if data_flag not in self.overlap_sets:
                pred = predictions[subset_id==i, base_cls:base_cls+ncls]
                y_scores[data_flag] = pred
                base_cls += ncls
            else:
                pred = predictions[subset_id==i, base_cls:base_cls+ncls]
                y_scores[data_flag] = pred

        # evaluate each subset
        metrics = {}
        for data_flag, pred in y_scores.items():
            task = medmnist.INFO[data_flag]["task"]
            if task == 'multi-label, binary-class':
                outputs = F.sigmoid(pred)
            else:
                outputs = F.softmax(pred, dim=1)
            evaluator = self.evaluators[data_flag]
            auc, acc = evaluator.evaluate(pred.detach().cpu().numpy(), None, None)
            metrics[f"{data_flag}_auc"] = auc
            metrics[f"{data_flag}_acc"] = acc

        pred = F.sigmoid(predictions).detach().cpu().numpy()
        auc, acc = self.mevaluator.evaluate(pred, None, None)
        metrics["overall_auc"] = auc
        metrics["overall_acc"] = acc

        return metrics