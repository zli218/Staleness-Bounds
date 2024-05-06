import ray
import logging
from net import ConvNet
from parameter_server import ParameterServer
from worker import Worker
from utils import get_data_loader, evaluate

    
def train(
    iterations,
    num_workers,
    lr,
    num_returns,
):
    model = ConvNet()
    test_loader = get_data_loader()[1]

    ps = ParameterServer.remote(lr)
    workers = [Worker.remote() for i in range(num_workers)]

    ps_clock = 0

    accuracy_list=[]

    print("Running Asynchronous Parameter Server Training.")

    current_weights = ps.get_weights.remote()
    worker_clocks = {}
    grad_refs = {}
    for worker in workers:
        grad_refs[worker.compute_gradients.remote(current_weights)] = worker
        worker_clocks[worker] = ps_clock

    for i in range(iterations):
        ready_grad_list, _ = ray.wait(
            list(grad_refs.keys()),
            num_returns=num_returns,
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
        print("Staleness: {}".format(staleness_list))
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

    iterations = 200
    num_workers = 32
    lr = 0.01
    num_returns = 32

    train(
        iterations=iterations,
        num_workers=num_workers,
        lr=lr,
        num_returns=num_returns,
    )

    ray.shutdown()
