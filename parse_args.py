import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--dataset",
                        type=str,
                        default = 'OpenBioLink',
                        # default = 'PrimeKG',
                        # default = 'BioKG',
                        help="dataset for training")

argparser.add_argument('--seed', type=int, default=1234, help='random seed')

argparser.add_argument('--num_layers', default=3, type=int, help='number of layers')

argparser.add_argument('--gpu', type=int, default=0,
                    help='gpu device')

argparser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

argparser.add_argument('--epochs', type=int, default=1000, help='training epoch')

argparser.add_argument('--batch_size', type=int, default=256, help='batch size')

argparser.add_argument('--patience', type=int, default = 80, help='early stopping patience')

argparser.add_argument('--weight_decay', type=float, default=1e-3, help='AdamW weight decay')

argparser.add_argument('--K_fold', type=int, default=10, help='k-fold cross validation')

argparser.add_argument('--validation_ratio', type=float, default=0.1, help='validation set partition ratio')

argparser.add_argument('--negative_rate', type=float, default=1.0,help='negative_rate')

argparser.add_argument('--negative_strategy', type=str, default="random", help='negative sampling strategy')

argparser.add_argument('--in_dim', default=64, type=int, help='input dimension')

argparser.add_argument('--hidden_dim', default=128, type=int, help='hidden dimension')

argparser.add_argument('--out_dim', default=64, type=int, help='output dimension')

argparser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')

argparser.add_argument('--dropout', default=0.2, type=float, help='dropout')

argparser.add_argument('--best_metric', type=str, default="Mix")

args = argparser.parse_args()