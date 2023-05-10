
import torchmetrics
from torchmetrics import MetricCollection

def classification_metrics(metric_list, num_classes, device):
    allowed_metrics = ['precision', 'recall', 'f1_score', 'accuracy']

    if metric_list:
        for metric in metric_list:
            if metric not in allowed_metrics:
                ValueError(f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1_score', 'accuracy'") 
        metric_dict = {
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'precision': torchmetrics.Precision(task='multiclass', average='macro', num_classes=num_classes).to(device),
            'recall': torchmetrics.Recall(task='multiclass', average='macro', num_classes=num_classes).to(device),
            'f1_score': torchmetrics.F1Score(task='multiclass', average='macro', num_classes=num_classes).to(device)
        }
    
    else:
        metric_list = ['accuracy']
        metric_dict = {
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device)}

   
    
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)