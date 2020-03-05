import argparse
#Parameter Configuration

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--min_loss', type=float, default=1e10,
                    help='min loss value')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='combination_ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio') 
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,#default = 100000
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='GCNConv/SAGEConv/ChebConv/GATConv/GraphConv')
parser.add_argument('--feature_fusion_type', type=str, default='GATConv',
                    help='GATConv')
parser.add_argument('--save_path', type=str, default='/home/GSAPool',
                    help='path to save result')
parser.add_argument('--training_times', type=int, default=20,
                    help='')



