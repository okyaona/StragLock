# StragLock/StragLock+

This is the Pytorch reproduction of StragLock/StragLock+. Experiments are produced on MNIST, HAM10000 and CIFAR10 datasets using LeNet, AlexNet, and ResNet-18.

## Requirments
 - Python3
 - Pytorch
 - Torchvision

## Data
 - Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
 - Experiments are run on MNIST, HAM10000 and CIFAR10 datasets.
 - To use your own dataset: Move your dataset under data/ directory.

## Running the experiments

There are many parameters involved, please feel free to play with option.py.

Here are several examples:
 - To run the experiment on MNIST with 5 clients including 1 malicious client launching random attacks:

   ```sh
   python main_handler_GPU_server.py --num_users 5 --malicious_client_number 1 --malicious_index 1 --dataset 'MNIST'
   
 - To run the experiment on CIFAR10 with 10 clients including 2 malicious clients launching history attacks for 100 epochs and 300 batch size:
   ```sh
   python main_handler_GPU_server.py --num_users 10 --malicious_client_number 2 --malicious_index 2 --dataset 'CIFAR10' --epochs 100 --batch_size 300

  ## Options
  The default values for various parameters parsed to the experiment are given in options.py. Details are given on some of those parameters:

 - `--num_users`: number of clients, default 2.
 - `--epochs`: number of epochs, default 10.
 - `--lr`: learning rate, default 0.001.
 - `--batch_size`: number of batch size, default 256.
 - `-malicious_client_number`: number of the malicious client, default 0.
