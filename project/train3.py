import ray
import logging
from net import ConvNet
from parameter_server import ParameterServer
from worker import Worker
from utils import get_data_loader, evaluate
from train import train
import matplotlib.pyplot as plt

    
def train3(
    iterations,
    num_workers,
    lr,
    num_returns,
    lr_decay,
    lower_returns
):
    model = ConvNet()
    test_loader = get_data_loader()[1]

    ps = ParameterServer.remote(lr)
    workers = [Worker.remote() for i in range(num_workers)]

    ps_clock = 0

    accuracy_list=[]

    print("Running Asynchronous Parameter Server Training.")
    print("Initial num_return: {}".format(num_returns))

    current_weights = ps.get_weights.remote()
    worker_clocks = {}
    grad_refs = {}
    orig_lr=lr
    for worker in workers:
        grad_refs[worker.compute_gradients.remote(current_weights)] = worker
        worker_clocks[worker] = ps_clock

    for i in range(iterations):
        lr=lr*lr_decay
        num_returns_decay=int(num_returns*(lr/orig_lr))
        if num_returns_decay<lower_returns:
            num_returns_decay=lower_returns
        ready_grad_list, _ = ray.wait(
            list(grad_refs.keys()),
            num_returns=num_returns_decay,
        )
        
        # Compute and apply gradients
        ready_grads = ray.get(ready_grad_list)
        current_weights = ps.apply_gradients.remote(ready_grads[0])
        ps_clock = ps_clock + 1
        
        # Compute staleness and update worker clocks
        staleness_list = []
        for ready_grad_id in ready_grad_list:
            worker = grad_refs.pop(ready_grad_id)
            staleness_list.append(ps_clock - 1 - worker_clocks[worker])

            grad_refs[worker.compute_gradients.remote(current_weights)] = worker
            worker_clocks[worker] = ps_clock

        # Evaluate the current model
        model.set_weights(ray.get(current_weights))
        accuracy = evaluate(model, test_loader)
        accuracy_list.append(accuracy)

        print("")
        print("Iter {}:".format(i))
        # print("Staleness: {}".format(staleness_list))
        print("Cuurent num_return: {}".format(num_returns_decay))
        print("Accuracy {}".format(accuracy))

    print("")
    print("***")
    print("Final accuracy is {:.1f}.".format(accuracy))
    print("***")

    return accuracy_list


if __name__ == "__main__":
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.INFO,
        include_dashboard=False,
    )

    iterations = 500
    num_workers = 32
    lr = 0.01
    num_returns = 32
    lr_decay= 0.997
    lower_returns=10

#     accuracy_list1=train3(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=num_returns, # this is just the initial num_returns in this code
#         lr_decay=lr_decay,
#         lower_returns=lower_returns
#     )
#
#     accuracy_list2=train(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=32,
#     )
#
#     accuracy_list3=train(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=1,
#     )
#
#     accuracy_list4=train(
#         iterations=iterations,
#         num_workers=num_workers,
#         lr=lr,
#         num_returns=8,
#     )
#
# accuracies_dict = {}
# accuracies_dict['dynamic_k'] = accuracy_list1
# accuracies_dict['sync_k8'] = accuracy_list4
# accuracies_dict['sync'] = accuracy_list2
# accuracies_dict['async'] = accuracy_list3
#
#
#
# def plot_accuracy(accuracies_dict,save_path='accuracy_comparison.png'):
#     plt.figure(figsize=(12, 6))
#     for label, accuracies in accuracies_dict.items():
#         plt.plot(accuracies, label=label)
#     plt.xlabel('Iteration')
#     plt.ylabel('Accuracy')
#     plt.title('Model Accuracy Comparison over Iterations')
#     plt.legend()
#     plt.grid(True)
#
#     plt.savefig(save_path, format='png', dpi=300)
#     plt.show()
#
#
# plot_accuracy(accuracies_dict)


ray.shutdown()
