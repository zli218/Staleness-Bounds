import ray
import logging
from net import ConvNet
from parameter_server import ParameterServer
from worker import Worker
from utils import get_data_loader, evaluate
from train import train
from train3 import train3
import matplotlib.pyplot as plt
import pickle


def plot_accuracy(accuracies_dict, save_path='accuracy_comparison6.png'):
    plt.figure(figsize=(12, 6))
    for label, accuracies in accuracies_dict.items():
        plt.plot(accuracies, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison over Iterations')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, format='png', dpi=300)
    plt.show()




iterations = 500
num_workers = 32
lr = 0.01
lr_decay= 0.997

# accuracy_list1=train3(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=32, # this is just the initial num_returns in this code
#         lr_decay=lr_decay,
#         lower_returns=8
#     )
#
# accuracy_list2=train3(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=16, # this is just the initial num_returns in this code
#         lr_decay=lr_decay,
#         lower_returns=8
#     )

# accuracy_list3=train3(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=32, # this is just the initial num_returns in this code
#         lr_decay=lr_decay,
#         lower_returns=4
#     )

# accuracy_list4=train3(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=16, # this is just the initial num_returns in this code
#         lr_decay=lr_decay,
#         lower_returns=2
#     )
# accuracy_list1=train(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=1,
#     )
# #
# accuracy_list2=train(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=8,
#     )
#
# accuracy_list3=train(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=16,
#     )

accuracy_list4=train(
        iterations=iterations,
        num_workers=num_workers,
        lr=lr,
        num_returns=32,
    )





accuracies_dict = {}
accuracies_dict['k=32'] = accuracy_list4
# accuracies_dict['k=8'] = accuracy_list2
# accuracies_dict['k=16'] = accuracy_list3
# accuracies_dict['k=32'] = accuracy_list4


# File path to save the dictionary
file_path = "ksyn_acc_k32.pkl"

# Save the dictionary to a file using Pickle
with open(file_path, "wb") as pickle_file:
    pickle.dump(accuracies_dict, pickle_file)

plot_accuracy(accuracies_dict)


ray.shutdown()