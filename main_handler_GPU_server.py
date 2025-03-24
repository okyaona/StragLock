import copy
import torch
from torch import nn
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from glob import glob
import random
import numpy as np
import time
from util.data_process import dataset_iid, SkinData, estimate_benign_updates
from util.other_utils import prRed, prGreen
from util.evaluation import calculate_accuracy
from util.SL_LeNet_util import LeNet_client_side, LeNet_server_side
from util.SL_AlexNet_util import AlexNet_client_side, AlexNet_server_side
from util.SL_ResNet_util import ResNet18_client_side, ResNet18_server_side, Baseblock
from util.attack_type_3 import poison_benign_updates_1
import option
import os


args = option.parse_opt()
print(args)
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print("cuda is available: ", torch.cuda.get_device_name(0))

torch.cuda.empty_cache()
torch.cuda.synchronize()

model_type = [0, 1, 2]
dic_to_csv = {}
hy_index = args.filename_index  # indicate output document name
history_num = args.history_num   # indicate how many updates participating in attacks
dataset = args.dataset   # used dataset: HAM, CIFAR10, MNIST
classes = args.classes
num_ep = args.num_ep

for j_index in model_type:
    num_users = args.num_users
    epochs = args.epochs
    frac = args.frac
    lr = args.lr
    local_ep_li = args.local_ep
    batch_size = args.batch_size
    # ==========================
    malicious_client_number = args.malicious_client_number
    client_update_lists = [[] for _ in range(malicious_client_number)]
    # ==========================
    malicious_index = args.malicious_index   #  {0: no attack, 1: random attack, 2: history attack, 3: model poisoning attack (NDSS)}
    alpha = args.alpha  # Scaling factor
    model_type = j_index  # {0: LeNet, 1: AlexNet, 2: ResNet}

    attack_time = list()
    attack_time_each_round = list()

    # =====================================================================================================
    #                                  Beginning
    # =====================================================================================================
    program = "SL on " + str(dataset)
    print(
        f"---------{program}, model_type {model_type}, malicious_index {malicious_index}, num_users {num_users}----------")  # this is to identify the program in the slurm outputs files
    # device = torch.device('cpu')  #  if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using: ", device)

    # =====================================================================================================
    #                           Client-side and Server-side Model
    # =====================================================================================================
    if model_type == 0:
        net_glob_client = LeNet_client_side()
        net_glob_server = LeNet_server_side(classes, dataset)
        print('get LeNet')

    if model_type == 1:
        net_glob_client = AlexNet_client_side()
        net_glob_server = AlexNet_server_side(dataset, classes)
        print('get AlexNet')

    if model_type == 2:
        net_glob_client = ResNet18_client_side()
        net_glob_server = ResNet18_server_side(Baseblock, [2, 2, 2], classes)
        print('get ResNet')

    net_glob_client.to(device)
    net_glob_server.to(device)

    # ===================================================================================
    # For Loss and Accuracy
    loss_train_collect = []
    acc_train_collect = []
    loss_test_collect = []
    acc_test_collect = []
    batch_acc_train = []
    batch_loss_train = []
    batch_acc_test = []
    batch_loss_test = []

    criterion = nn.CrossEntropyLoss()
    count1 = 0
    count2 = 0

    # ====================================================================================================
    #                                  Server Side Program
    # ====================================================================================================
    # to print train - test together in each round-- these are made global
    acc_avg_all_user_train = 0
    loss_avg_all_user_train = 0
    loss_train_collect_user = []
    acc_train_collect_user = []
    loss_test_collect_user = []
    acc_test_collect_user = []
    idx_collect = []
    l_epoch_check = False
    fed_check = False

    # Server-side function associated with Training
    def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
        global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
        global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
        global loss_train_collect_user, acc_train_collect_user

        net_glob_server.train()
        optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

        # train and update
        optimizer_server.zero_grad()

        fx_client = fx_client.to(device)
        y = y.to(device)

        # ---------forward prop-------------
        fx_server = net_glob_server(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        # --------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        optimizer_server.step()

        batch_loss_train.append(loss.item())
        batch_acc_train.append(acc.item())

        count1 += 1
        if count1 == len_batch:
            acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)
            loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

            batch_acc_train = []
            batch_loss_train = []
            count1 = 0

            prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count,
                                                                                          acc_avg_train,
                                                                                          loss_avg_train))
            if l_epoch_count == l_epoch - 1:

                l_epoch_check = True
                acc_avg_train_all = acc_avg_train
                loss_avg_train_all = loss_avg_train
                loss_train_collect_user.append(loss_avg_train_all)
                acc_train_collect_user.append(acc_avg_train_all)
                if idx not in idx_collect:
                    idx_collect.append(idx)

            if len(idx_collect) == num_users:
                fed_check = True
                idx_collect = []
                acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
                loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
                loss_train_collect.append(loss_avg_all_user_train)
                acc_train_collect.append(acc_avg_all_user_train)
                acc_train_collect_user = []
                loss_train_collect_user = []
        return dfx_client

    # Server-side functions associated with Testing
    def evaluate_server(fx_client, y, idx, len_batch, ell):
        global net_glob_server, criterion, batch_acc_test, batch_loss_test
        global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
        global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

        net_glob_server.eval()

        with torch.no_grad():
            fx_client = fx_client.to(device)
            y = y.to(device)
            # ---------forward prop-------------
            fx_server = net_glob_server(fx_client)

            # calculate loss
            loss = criterion(fx_server, y)
            # calculate accuracy
            acc = calculate_accuracy(fx_server, y)

            batch_loss_test.append(loss.item())
            batch_acc_test.append(acc.item())

            count2 += 1
            if count2 == len_batch:
                acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
                loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

                batch_acc_test = []
                batch_loss_test = []
                count2 = 0

                prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                                 loss_avg_test))
                if l_epoch_check:
                    l_epoch_check = False
                    # Store the last accuracy and loss
                    acc_avg_test_all = acc_avg_test
                    loss_avg_test_all = loss_avg_test
                    loss_test_collect_user.append(loss_avg_test_all)
                    acc_test_collect_user.append(acc_avg_test_all)
                if fed_check:
                    fed_check = False

                    acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                    loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                    loss_test_collect.append(loss_avg_all_user)
                    acc_test_collect.append(acc_avg_all_user)
                    acc_test_collect_user = []
                    loss_test_collect_user = []

                    print("====================== SERVER V1==========================")
                    print(
                        ' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                            loss_avg_all_user_train))
                    print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                             loss_avg_all_user))
                    print("==========================================================")

        return


    # ==============================================================================================================
    #                                       Clients Side Program
    # ==============================================================================================================
    class DatasetSplit(Dataset):
        def __init__(self, dataset, idxs):
            self.dataset = dataset
            self.idxs = list(idxs)

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, item):
            image, label = self.dataset[self.idxs[item]]
            return image, label

    # Client-side functions associated with Training and Testing
    class Client(object):
        def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                     idxs_test=None, malicious_index=0):
            self.idx = idx
            self.device = device
            self.lr = lr
            self.local_ep = random.choice(local_ep_li)
            self.malicious_index = malicious_index
            self.batch_size = batch_size
            self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=batch_size, shuffle=True)
            self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=batch_size, shuffle=True)

        def train(self, net, malicious_index, epoch_iter, global_update_history):
            net.train()
            optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
            epoch_t_s = list()
            for iter in range(self.local_ep):
                len_batch = len(self.ldr_train)
                count = 0
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer_client.zero_grad()

                    # ---------forward prop-------------
                    fx = net(images)
                    total_start = time.time()

                    if malicious_index == 0:
                        client_fx = fx.clone().detach().requires_grad_(True)
                        global_update_history.append(client_fx)

                    if malicious_index == 1:
                        tmp_fx = fx.clone().detach()
                        shape = tmp_fx.shape
                        gaussian_noise_tensor = -10000 * tmp_fx - 100000 * torch.randn(shape, device=tmp_fx.device)
                        fake_tensor = gaussian_noise_tensor.detach().requires_grad_(True)
                        client_fx = fake_tensor

                    if malicious_index == 2:
                        client_fx = fx.clone().detach()
                        updated_client_fx = client_fx.clone().detach()
                        if len(global_update_history) > history_num:
                            global_update_history = global_update_history[-history_num:]
                        fake_update = estimate_benign_updates(global_update_history, alpha, updated_client_fx.size(0))
                        client_fx = fake_update.clone().detach().requires_grad_(True)
                        global_update_history.append(client_fx)

                    if malicious_index == 3:
                        client_fx = fx.clone().detach()
                        updated_client_fx = client_fx.clone()
                        if len(global_update_history) > history_num:
                            global_update_history = global_update_history[-history_num:]
                        poison_update = poison_benign_updates_1(global_update_history, updated_client_fx.size(0))
                        client_fx = poison_update.detach().requires_grad_(True)
                        global_update_history.append(client_fx)

                    total_end = time.time()
                    epoch_t_s.append(total_end - total_start)
                    attack_time.append(total_end - total_start)
                    # Sending activations to server and receiving gradients from server
                    dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                    # --------backward prop -------------
                    fx.backward(dfx)
                    optimizer_client.step()
                    count += 1
            tmp_m = {}
            tmp_m[malicious_index] = sum(epoch_t_s)
            attack_time_each_round.append(tmp_m)
            return net.state_dict()

        def evaluate(self, net, ell):
            net.eval()
            with torch.no_grad():
                len_batch = len(self.ldr_test)
                for batch_idx, (images, labels) in enumerate(self.ldr_test):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # ---------forward prop-------------
                    fx = net(images)
                    evaluate_server(fx, labels, self.idx, len_batch, ell)
            return

    # =============================================================================
    #                         Data loading
    # =============================================================================
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if dataset == 'HAM':
        df = pd.read_csv('data/HAM10000_metadata.csv')
        lesion_type = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join("data", '*', '*.jpg'))}
        df['path'] = df['image_id'].map(imageid_path.get)
        df['cell_type'] = df['dx'].map(lesion_type.get)
        df['target'] = pd.Categorical(df['cell_type']).codes

        # Train-test split
        train, test = train_test_split(df, test_size=0.2)
        train = train.reset_index()
        test = test.reset_index()

        # =============================================================================
        #                         Data preprocessing: Transformation
        # =============================================================================
        if model_type == 1 or model_type == 2:
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.RandomVerticalFlip(),
                                                   transforms.Pad(3),
                                                   transforms.RandomRotation(10),
                                                   transforms.CenterCrop(64),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std)
                                                   ])

            test_transforms = transforms.Compose([
                transforms.Pad(3),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        if model_type == 0:
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Pad(3),
                transforms.Resize((32, 32)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_transforms = transforms.Compose([
                transforms.Pad(3),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        dataset_train = SkinData(train, transform=train_transforms)
        dataset_test = SkinData(test, transform=test_transforms)

    if dataset == 'MNIST':
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Pad(3),
            transforms.RandomRotation(10),
            transforms.CenterCrop(64),
            transforms.ToTensor(),  # Convert to Tensor first
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the tensor to 3 channels
            transforms.Normalize(mean=mean, std=std)  # Match 3 channels for normalization
        ])

        test_transforms = transforms.Compose([
            transforms.Pad(3),
            transforms.CenterCrop(64),
            transforms.ToTensor(),  # Convert to Tensor first
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the tensor to 3 channels
            transforms.Normalize(mean=mean, std=std)  # Match 3 channels for normalization
        ])
        dataset_train = datasets.MNIST(root='./data', train=True, transform=train_transforms, download=True)
        dataset_test = datasets.MNIST(root='./data', train=False, transform=test_transforms, download=True)

    if dataset == 'CIFAR10':
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.RandomVerticalFlip(),  # Randomly flip images vertically
            transforms.Pad(3),  # Pad the image with 4 pixels on each side
            transforms.RandomRotation(10),  # Randomly rotate the image by 10 degrees
            transforms.CenterCrop(32),  # Crop the image to a square of size 32x32
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=mean, std=std)  # Normalize with mean and std for RGB channels
        ])

        test_transforms = transforms.Compose([
            transforms.Pad(3),  # Pad the image with 4 pixels on each side
            transforms.CenterCrop(32),  # Crop the image to 32x32
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=mean, std=std) # Normalize with mean and std for RGB channels
        ])
        dataset_train = datasets.CIFAR10(root='./data', train=True, transform=train_transforms, download=True)
        dataset_test = datasets.CIFAR10(root='./data', train=False, transform=test_transforms, download=True)

    dict_users = dataset_iid(dataset_train, num_users)
    dict_users_test = dataset_iid(dataset_test, num_users)

    start = time.time()
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)  # fixed user list
    for iter in range(epochs):
        for idx in idxs_users[malicious_client_number:]:
            print("This is honest client: ", idx)
            local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                           idxs=dict_users[idx], idxs_test=dict_users_test[idx], malicious_index=0)
            # ------------------ Honest Training and Testing ------------------
            w_client = local.train(net=copy.deepcopy(net_glob_client).to(device), malicious_index=0, epoch_iter=iter,
                                   global_update_history=[])
            local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
            net_glob_client.load_state_dict(w_client)
        for idx in idxs_users[:malicious_client_number]:
            print("This is malicious client: ", idx)
            malicious_locations = np.where(idxs_users == idx)[0][0]
            global_update_history = client_update_lists[malicious_locations]
            local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                           idxs=dict_users[idx], idxs_test=dict_users_test[idx], malicious_index=malicious_index)
            # ------------------ Malicious Training and Testing ------------------
            if iter == num_ep - 1:
                w_client = local.train(net=copy.deepcopy(net_glob_client).to(device), malicious_index=0,
                                       epoch_iter=iter, global_update_history=global_update_history)
            else:
                w_client = local.train(net=copy.deepcopy(net_glob_client).to(device), malicious_index=malicious_index,
                                       epoch_iter=iter, global_update_history=global_update_history)
            local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
            net_glob_client.load_state_dict(w_client)
    end = time.time()
    print("Training and Evaluation completed!")
    print('total training time: ', end - start)
    print('including additional attacking time: ', sum(attack_time))
    print('no attackers training time: ', end - start - sum(attack_time))
