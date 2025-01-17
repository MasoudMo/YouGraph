import random
import torch
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
import numpy as np
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import datetime
from torch_geometric.utils import degree

import sys
sys.path.append('../..')

from model import Net, VirtualnodeNet
from utils.config import process_config, get_args
from utils.util import warm_up_lr, train_with_flag

class In:
    def readline(self):
        return "y\n"

    def close(self):
        pass


def train(model, device, loader, optimizer, multicls_criterion):
    model.train()
    loss_all = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))

            loss.backward()
            loss_all += loss.item()
            optimizer.step()

    return loss_all / len(loader)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def dropgnn_config(dataset, p, num_runs):
    # Calculate the required properties if DropGNN is used
    # This is the implementation from https://arxiv.org/abs/2111.06283

    n = list()
    degs = list()

    for g in dataset:
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        n.append(g.num_nodes)
        degs.append(deg.max())

    # Compute and print the statistics of the dataset
    mean_deg = torch.stack(degs).float().mean()
    print("DropGNN - Mean degree: {}".format(mean_deg))

    max_deg = torch.stack(degs).max()
    print("DropGNN - Max degree: {}".format(max_deg))

    min_deg = torch.stack(degs).min()
    print("DropGNN - Min degree: {}".format(min_deg))

    mean_n = torch.tensor(n).float().mean().round().long().item()
    print("DropGNN - Mean number of nodes: {}".format(mean_n))

    max_n = torch.tensor(n).float().max().round().long().item()
    print("DropGNN - Max number of nodes: {}".format(max_n))

    min_n = torch.tensor(n).float().min().round().long().item()
    print("DropGNN - Min number of nodes: {}".format(min_n))

    gamma = mean_n

    if p is None:
        p = 2 * 1 / (1 + gamma)
        print("DropGNN - Recommended p: {}".format(p))

    if num_runs is None:
        num_runs = gamma
        print("DropGNN - Recommended Number of Runs: {}".format(num_runs))

    print("DropGNN - Chosen Number of Runs: {}".format(num_runs))
    print("DropGNN - Chosen Sampling Probability: {}".format(p))

    return p, num_runs


def main():
    args = get_args()
    config = process_config(args)
    print(config)

    #if config.get('seed') is not None:
    #    random.seed(config.seed)
    #    torch.manual_seed(config.seed)
    #    np.random.seed(config.seed)
    #    if torch.cuda.is_available():
    #        torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name=config.dataset_name, transform=add_zeros, root=config.dataset_root)

    split_idx = dataset.get_idx_split()

    # Compute DropGNN paramteres if needed
    dropgnn_p = 0.0
    dropgnn_num_runs = 1
    if config.use_dropgnn:
        dropgnn_p, dropgnn_num_runs = dropgnn_config(dataset[split_idx["train"]],
                                                     config.dropgnn_p,
                                                     config.dropgnn_num_runs)

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(config.dataset_name)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config.hyperparams.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]],
                              batch_size=config.hyperparams.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]],
                             batch_size=config.hyperparams.batch_size, shuffle=False, num_workers=config.num_workers)

    if config.architecture.virtual_node == 'true':
        model = VirtualnodeNet(config.architecture,
                               num_class=dataset.num_classes,
                               drop_gnn=config.use_dropgnn,
                               node_dropout_p=dropgnn_p,
                               num_runs=dropgnn_num_runs,
                               nodeskip=config.use_nodeskip,
                               nodeskip_dropout_p=config.nodeskip_p).to(device)
    else:
        model = Net(config.architecture,
                    num_class=dataset.num_classes,
                    drop_gnn=config.use_dropgnn,
                    node_dropout_p=dropgnn_p,
                    num_runs=dropgnn_num_runs,
                    nodeskip=config.use_nodeskip,
                    nodeskip_dropout_p=config.nodeskip_p).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    
    optimizer = optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate, weight_decay=config.hyperparams.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.hyperparams.step_size,
    #                                            gamma=config.hyperparams.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.hyperparams.epochs - config.hyperparams.warmup_epochs)

    multicls_criterion = torch.nn.CrossEntropyLoss()

    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []


    save_file = args.dir + '/' + str(config.time_stamp) + '_' \
                    + 'S' + str(config.seed if config.get('seed') is not None else "na") + '.pt'
    
    train_epoch = train_with_flag if config.architecture.flag == 'true' else train

    for epoch in range(1, config.hyperparams.epochs + 1):
        if epoch <= config.hyperparams.warmup_epochs:
            warm_up_lr(epoch, config.hyperparams.warmup_epochs, config.hyperparams.learning_rate, optimizer)
        print (datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        print("Epoch {} training...".format(epoch))
        print ("lr: ", optimizer.param_groups[0]['lr'])
        
        train_loss = train_epoch(model, device, train_loader, optimizer, multicls_criterion)
        if epoch > config.hyperparams.warmup_epochs:
            scheduler.step()


        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print('Train:', train_perf[dataset.eval_metric],
              'Validation:', valid_perf[dataset.eval_metric],
              'Test:', test_perf[dataset.eval_metric],
              'Train loss:', train_loss)

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        trainL_curve.append(train_loss)

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)

    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    print('Finished test: {}, Validation: {}, Train: {}, epoch: {}, best train: {}, best loss: {}'
          .format(test_curve[best_val_epoch], valid_curve[best_val_epoch], train_curve[best_val_epoch],
                  best_val_epoch, best_train, min(trainL_curve)))
    if config.save == 'true':
        torch.save(model.state_dict(), save_file)

if __name__ == "__main__":
    main()
