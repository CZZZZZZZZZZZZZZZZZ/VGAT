import argparse



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument(
        "--data_root_dir", type=str, default="", help="Data directory to WSI features (extracted via CLAM"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible experiment (default: 1)")
    parser.add_argument(
        "--which_splits", type=str, default="5fold", help="Which splits folder to use in ./splits/ (Default: ./splits/5foldcv"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tcga_blca",
        help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)',
    )
    parser.add_argument("--log_data", action="store_true", default=True, help="Log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", default=False,dest="evaluate", help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="None", metavar="PATH", help="Path to latest checkpoint (default: none)")
    parser.add_argument("--OOM", type=int, default=4096, help="Ramdomly sampling some patches to avoid OOM error")
    # Model Parameters.
    parser.add_argument(
        "--model",
        type=str,
        default="vgat",
        help="Type of model (Default: mcat)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=[
            "small",
            "large",
        ],
        default="small",
        help="Size of some models (Transformer)",
    )
    parser.add_argument(
        "--modal",
        type=str,
        choices=["omic", "path", "coattn","bert_coattn"],
        default="bert_coattn",
        help="Specifies which modalities to use / collate function in dataloader.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "bilinear"],
        default="concat",
        help="Modality fuison strategy",
    )
    

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam",
                        "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--num_epoch", type=int, default=100, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--loss",
        type=str,
        default="nll_surv_kl",
        help="slide-level classification loss function",
    )
    parser.add_argument("--weighted_sample", action="store_true", default=False, help="Enable weighted sampling")
    parser.add_argument("--select", default='em', choices=["em", "cluster", "rand"],help="whether to select patch")
    parser.add_argument("--save_risk", default=False, help="whether to select patch")
    parser.add_argument("--proto_name", default='prototypes_c16_resnet50_20_faiss_num_1.0e+06.pkl', help="Cluster center vector file name")
    
    args = parser.parse_args()
    return args
