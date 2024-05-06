import matplotlib.pyplot as plt
import pickle


def plot_accuracy(accuracies_dict, save_path='acc.png'):
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

def load_dictionary_from_pickle(file_path):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



file_path='final_ksyn_acc2.pkl'
ksyn = load_dictionary_from_pickle(file_path)


accuracies_dict = ksyn


plot_accuracy(accuracies_dict,save_path='ksyn_acc2.png')

