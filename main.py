
import os
import dgl
import torch
import random
import numpy as np
from parse_args import args
from hypergraph_data import HypergraphDatasetDGL
from dataloader import DrugDiseaseKFoldDataset
from model import DrugRepositioningModel
from model_train import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)

def train_and_evaluate(device, log_folder_path):
    hypergraphDataset = HypergraphDatasetDGL(device = device)
    hypergraph = hypergraphDataset.g

    kfold_dataset = DrugDiseaseKFoldDataset(hypergraphDataset)

    l_test_fold_results = []
    for train_loader, val_loader, test_loader, fold_i in kfold_dataset.get_fold_dataloaders():
        print("Training Fold {}".format(fold_i))
        
        model = DrugRepositioningModel(hypergraph = hypergraph, device= device, in_dim = args.in_dim, hidden_dim = args.hidden_dim, out_dim = args.out_dim, num_heads = args.num_heads, num_layers = args.num_layers, dropout = args.dropout)
        model = model.to(device)

        trainer = Trainer(model, device, log_folder_path)

        trainer.train(train_loader, val_loader, fold_i)

        out_path = os.path.join(log_folder_path, "fold_{}".format(fold_i))
        ckpt_path = os.path.join(out_path, "result.ckpt")

        trainer.model.load(ckpt_path)

        test_results_fold_i = trainer.evaluate(test_loader)

        l_test_fold_results.append(test_results_fold_i)
    
    np_results = np.array(l_test_fold_results)
    mean_results = np.mean(np_results, axis=0)
    print(mean_results)

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    if use_cuda and args.gpu < torch.cuda.device_count():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    set_seed(args.seed)

    current_dir = os.getcwd()
    log_folder_path = os.path.join(current_dir, "Logs_/{}".format(args.dataset))
    
    train_and_evaluate(device, log_folder_path)
