import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=2, help='number of clients')
    parser.add_argument('--filename_index', type=str, default='', help='filename index')
    parser.add_argument('--seed', type=int, default=1234, help='number of random seed')
    parser.add_argument('--frac', type=float, default=1.0, help='participation of clients')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--local_ep', nargs="+", type=int, default=1, help='list of the number of local epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='number of batch size')
    parser.add_argument('--malicious_client_number', type=int, default=0, help='malicious client number')
    parser.add_argument('--malicious_index', type=int, default=1, help='attack type')  #  {0: no attack, 1: random attack, 2: history attack, 3: model poisoning attack 1 (NDSS'21)}
    parser.add_argument('--alpha', type=float, default=-150, help='learning rate')  # Scaling factor
    parser.add_argument('--history_num', type=int, default=20, help='number of iteration in the dataset attack')
    parser.add_argument('--dataset', type=str, default='HAM', help='dataset name: MNIST, CIFAR10, HAM')
    parser.add_argument('--num_ep', type=int, default=1, help='number of global epochs during which malicious clients don not attack')

    if parser.parse_args().dataset == 'HAM':
        parser.add_argument('--data_path', type=str, default='data/HAM10000_metadata.csv', help='path of dataset')
        parser.add_argument('--chan', type=int, default=3, help='number of image channel')
        parser.add_argument('--classes', type=int, default=7, help='number of classes')
    elif parser.parse_args().dataset == 'CIFAR10':
        parser.add_argument('--chan', type=int, default=3, help='number of image channel')
        parser.add_argument('--classes', type=int, default=10, help='number of classes')
    elif parser.parse_args().dataset == 'MNIST':
        parser.add_argument('--chan', type=int, default=1, help='number of image channel')
        parser.add_argument('--classes', type=int, default=10, help='number of classes')
    args = parser.parse_args()

    return args