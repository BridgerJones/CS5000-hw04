#/usr/bin/python

from ann import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

####################################
# CS5600/6600: F21: HW04
# Bridger Jones
# A02314787
# Write your code at the end of
# this file in the provided
# function stubs.
#####################################

#### auxiliary functions
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of ann.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = ann(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### plotting costs and accuracies
def plot_costs(eval_costs, train_costs, num_epochs):
    plt.title('Evaluation Cost (EC) and Training Cost (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_costs, label='EC', c='g')
    plt.plot(epochs, train_costs, label='TC', c='b')
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc='best')
    plt.show()

def plot_accuracies(eval_accs, train_accs, num_epochs):
    plt.title('Evaluation Acc (EA) and Training Acc (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_accs, label='EA', c='g')
    plt.plot(epochs, train_accs, label='TA', c='b')
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc='best')
    plt.show()

## num_nodes -> (eval_cost, eval_acc, train_cost, train_acc)
## use this function to compute the eval_acc and min_cost.
# This method builds a network that is 784XnX10 where 10 <= n <= 11
# The best performance was 784x11x10 at 30 epochs with the defaults in the unit tests.
# Cost on training data: 0.4217011631793625
# Accuracy on training data: 46838 / 50000
# Cost on evaluation data: 0.500813220704175
# Accuracy on evaluation data: 9234 / 10000
def collect_1_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    ### your code here
    stats = {}
    best_net = None
    best_stats = None
    for i in range(lower_num_hidden_nodes,upper_num_hidden_nodes + 1):
        print("LOOP", i)
        net = ann([784,i,10], cost=cost_function)
        sgd_results = net.mini_batch_sgd(train_data,
        num_epochs,mbs,eta, lmbda=lmbda, evaluation_data=eval_data,
                                        monitor_evaluation_cost=True,
                                        monitor_evaluation_accuracy=True,
                                        monitor_training_cost=True,
                                        monitor_training_accuracy=True)
        print(sgd_results)
        stats[i] = sgd_results
        # determine if network is worth saving
        if best_net == None:
            best_net = net
            best_stats = sgd_results
        else:
            if (np.mean(sgd_results[2]) < np.mean(best_stats[2]) and np.mean(sgd_results[3]) > np.mean(best_stats[3])):
                best_stats = sgd_results
                best_net = net
    print("LEN OF DICT", len(stats))
    print("BEST NET", best_stats)
    best_net.save("net1.json")
    plot_accuracies(best_stats[1], best_stats[3], num_epochs)
    plot_costs(best_stats[0], best_stats[2], num_epochs)
    return stats


# This method builds a network that is 784XnXnX10 where 2 <= n <= 3
# The best performance was 784x3x3x10 at 30 epochs with the defaults in the unit tests.
# net2's accuracy on evaluation data: 6468 / 10000
def collect_2_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    ### your code here
    stats = {}
    best_net = None
    best_stats = None
    for i in range(lower_num_hidden_nodes,upper_num_hidden_nodes + 1):
        for j in range(lower_num_hidden_nodes,upper_num_hidden_nodes + 1):
            print("LOOP", i)
            net = ann([784,i,j,10], cost=cost_function)
            sgd_results = net.mini_batch_sgd(train_data,
            num_epochs,mbs,eta, lmbda=lmbda, evaluation_data=eval_data,
                                            monitor_evaluation_cost=True,
                                            monitor_evaluation_accuracy=True,
                                            monitor_training_cost=True,
                                            monitor_training_accuracy=True)
            print(sgd_results)
            stats[f"{i}_{j}"] = sgd_results
            # determine if network is worth saving
            if best_net == None:
                best_net = net
                best_stats = sgd_results
            else:
                if (np.mean(sgd_results[2]) < np.mean(best_stats[2]) and np.mean(sgd_results[3]) > np.mean(best_stats[3])):
                    best_stats = sgd_results
                    best_net = net
    print("LEN OF DICT", len(stats))
    print("BEST NET", best_stats)
    best_net.save("net2.json")
    plot_accuracies(best_stats[1], best_stats[3], num_epochs)
    plot_costs(best_stats[0], best_stats[2], num_epochs)
    return stats


# This method builds a network that is 784XnXnXnX10 where 2 <= n <= 3
# The best performance was 784x3x2x3x10 at 30 epochs with the defaults in the unit tests.
# net3's accuracy on evaluation data: 6197 / 10000

def collect_3_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    ### your code here
        stats = {}
        best_net = None
        best_stats = None
        for i in range(lower_num_hidden_nodes,upper_num_hidden_nodes + 1):
            for j in range(lower_num_hidden_nodes,upper_num_hidden_nodes + 1):
                for k in range(lower_num_hidden_nodes,upper_num_hidden_nodes + 1):
                    print("LOOP", i,j,k)
                    net = ann([784,i,j,k,10], cost=cost_function)
                    sgd_results = net.mini_batch_sgd(train_data,
                    num_epochs,mbs,eta, lmbda=lmbda, evaluation_data=eval_data,
                                                    monitor_evaluation_cost=True,
                                                    monitor_evaluation_accuracy=True,
                                                    monitor_training_cost=True,
                                                    monitor_training_accuracy=True)
                    print(sgd_results)
                    stats[f"{i}_{j}_{k}"] = sgd_results
                    # determine if network is worth saving
                    if best_net == None:
                        best_net = net
                        best_stats = sgd_results
                    else:
                        if (np.mean(sgd_results[2]) < np.mean(best_stats[2]) and np.mean(sgd_results[3]) > np.mean(best_stats[3])):
                            best_stats = sgd_results
                            best_net = net
        print("LEN OF DICT", len(stats))
        print("BEST NET", best_stats)
        best_net.save("net3.json")
        plot_accuracies(best_stats[1], best_stats[3], num_epochs)
        plot_costs(best_stats[0], best_stats[2], num_epochs)
        return stats
