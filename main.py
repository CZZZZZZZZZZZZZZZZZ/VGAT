import os

import csv
import time
import numpy as np

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed

from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)

    # create results directory
    results_dir = os.path.join('./results', args.model, args.dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    # start 5-fold CV evaluation.
    for fold in range(5):
        # build dataset
        image_dir = os.path.join(args.data_root_dir, args.dataset,'feats_pt/')
        gene_dir = os.path.join(args.data_root_dir, args.dataset,'gene_pt/')
        dataset = Generic_MIL_Survival_Dataset(
            csv_path= './csv/gene_embedding_csv/'+args.dataset+'.csv',
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=None,
            data_dir=image_dir,
            Bert_data_dir=gene_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )
        split_dir = os.path.join('./splits',args.which_splits, args.dataset)
        print(split_dir)
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(split_dir, fold),Normalizing_Data=None
        )
        print("{}/splits_{}.csv".format(split_dir, fold))
        print(train_dataset)
        train_loader = get_split_loader(
            train_dataset,
            training=True,
            weighted=args.weighted_sample,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        val_loader = get_split_loader(
            val_dataset, modal=args.modal, batch_size=args.batch_size
        )
        print(
            "training: {}, validation: {}".format(len(train_dataset), len(val_dataset))
        )

        # build model, criterion, optimizer, schedular
        
        if  args.model == "vgat":
            from models.VGAT.network import VGAT
            from models.VGAT.engine import Engine
            proto_path = os.path.join(split_dir, "panther",str(fold),'prototypes',args.proto_name)
            model_dict = {
                "n_classes": 4,
                "fusion": args.fusion,
                "model_size": args.model_size,
                 "proto_path": proto_path,
                 "num" :int(args.OOM/4),
                 "select": args.select,
            }
            model = VGAT(**model_dict)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)

        
        else:
            raise NotImplementedError(
                "Model [{}] is not implemented".format(args.model)
            )
       
        import pickle
        score, epoch ,risk= engine.learning(
        model, train_loader, val_loader, criterion, optimizer, scheduler
            # save best score and epoch for each fold
        )
        if args.save_risk:
            path = ''
            path = os.path.join(path,str(fold),'risk.pkl')
            with open(path, 'wb') as file:
                pickle.dump(risk, file)
        best_epoch.append(epoch)
        best_score.append(score)
        

    # finish training
    # mean and std
    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(np.mean(best_score[1:6]))
    best_score.append(np.std(best_score[1:6]))

    csv_path = os.path.join(results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
