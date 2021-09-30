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
    return stats

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
    pass

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
    pass
