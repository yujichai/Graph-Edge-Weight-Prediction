"""
    experiment for
    leave N% out
"""
import numpy as np
import pandas as pd
from src.utils import *
from src.page_rank import Page_Rank
from src.bias_deserve import Bias_Deserve
from src.fairness_goodness import Fairness_Goodness
from src.reciprocal import Reciprocal
from src.signed_hits import Sighed_Hits
from src.status_theory import Status_Theory
from src.triadic_balance import Triadic_Balance
from src.triadic_status import Triadic_Status
from src.multiple_regression import Linear_Regression

from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset

def ew_take_average(es, ws):
    # Take average for repeatting edges
    dic = {}
    for i, row in enumerate(es):
        x = row[0]
        y = row[1]
        w = ws[i]
        if (x,y) in dic.keys():
            dic[(x,y)] = (dic[(x,y)][0]+w, dic[(x,y)][1]+1)
        else :
            dic[(x,y)] = (w, 1)
    
    keys = dic.keys()
    length = len(keys)
    edges = np.zeros((length, 2))
    weights = np.zeros(length)

    for i, key in enumerate(keys):
        edges[i][0] = key[0].astype(int)
        edges[i][1] = key[1].astype(int)
        weights[i] = dic[key][0] / dic[key][1]
    return edges, weights


def init_dataset(datasetName):
    """
    :param filename: Network.csv, path
    :return: nx.DiGraph()
    """
    dataset = PygLinkPropPredDataset(name=datasetName)
    split_edge = dataset.get_edge_split()
    
    # conbine train and validation 
    train_edge = split_edge['train']['edge'].numpy()
    train_weight = split_edge['train']['weight'].numpy()
    valid_edge = split_edge['valid']['edge'].numpy()
    valid_weight = split_edge['valid']['weight'].numpy()
    train_edge = np.concatenate((train_edge, valid_edge))
    train_weight = np.concatenate((train_weight, valid_weight))
    # load test
    test_edge = split_edge['test']['edge'].numpy()
    test_weight = split_edge['test']['weight'].numpy()

    # Add in neg edge
    neg_edge = split_edge['test']['edge_neg'].numpy()
    neg_weight = np.zeros(neg_edge.shape[0])

    # Create the edge + weight
    train_edge, train_weight = ew_take_average(train_edge, train_weight)
    test_edge, test_weight = ew_take_average(test_edge, test_weight)
    neg_edge, neg_weight = ew_take_average(neg_edge, neg_weight)

    return train_edge, train_weight, test_edge, test_weight, neg_edge, neg_weight

def init_graph_oglb(edgeTrain, weightTrain, rateTrain, edgeTest, weightTest, rateTest, edgeNeg, weightNeg):
    selectTrain = int(weightTrain.shape[0] * rateTrain)
    selectTest = int(weightTest.shape[0] * rateTest)
    selectNegTrain = int(weightNeg.shape[0] * rateTrain)
    selectNegTest = int(weightNeg.shape[0] * rateTest)

    # Shuffle all the ews
    ewTrain = np.concatenate((edgeTrain, weightTrain.reshape((-1, 1))), axis=1)
    ewTest = np.concatenate((edgeTest, weightTest.reshape((-1, 1))), axis=1)
    ewNeg = np.concatenate((edgeNeg, weightNeg.reshape((-1, 1))), axis=1)
    np.random.shuffle(ewTrain)
    np.random.shuffle(ewTest)
    np.random.shuffle(ewNeg)

    # Select a portion of the train and test
    ewTrainSelect = ewTrain[0:selectTrain, :]
    ewTestSelect = ewTest[0:selectTest, :]
    ewNegSelect = ewNeg[0:selectNegTrain+selectNegTest, :]
    ewNegTestSelect = ewNeg[0:selectNegTest, :]
    # Create the total graph with all the edges and weights
    ewTotalSelect = np.concatenate((ewTrainSelect, ewTestSelect))
    ewTotalSelect = np.concatenate((ewTotalSelect, ewNegSelect))
    ewTotalTestSelect = np.concatenate((ewTestSelect, ewNegTestSelect))
    edgeTotalSelect, weightTotalSelect = ew_take_average(ewTotalSelect[:,0:2], ewTotalSelect[:,2])

    GTotal = nx.DiGraph()
    for i, e in enumerate(edgeTotalSelect):
        GTotal.add_edge(e[0], e[1], weight=weightTotalSelect[i], signed_weight=1, positive=weightTotalSelect[i], negative=0)

    GTrain = GTotal.copy()
    GTrain.remove_edges_from(ewTotalTestSelect[:,0:2])

    return GTotal, GTrain

def run_algorithm(algorithmName, GTrain):
    result = None
    if algorithmName=='pr':
        result = Page_Rank(GTrain)
    elif algorithmName=='bd':
        result = Bias_Deserve(GTrain)
    elif algorithmName=='fg':
        result = Fairness_Goodness(GTrain)
    elif algorithmName=='rp':
        result = Reciprocal(GTrain)
    elif algorithmName=='sh':
        result = Sighed_Hits(GTrain)
    elif algorithmName=='st':
        result = Status_Theory(GTrain)
    elif algorithmName=='tb':
        result = Triadic_Balance(GTrain)
    elif algorithmName=='ts':
        result = Triadic_Status(GTrain)
    else:
        raise NameError('Algorithm Name Not Found!')

    return result

def main():
    # Set random seed
    np.random.seed(42) 

    print('\nSelect dataset for evaluation\n')
    print('Avaliable datset: ogbl-collab\n')

    datasetName = 'ogbl-collab'

    trainEdge, trainWeight, testEdge, testWeight, negEdge, negWeight = init_dataset(datasetName)
    trainRates = [0.1]
    testRate = 0.05

    algorithm_type = ['PageRank', 'Bias_Deserve', 'Fairness_Goodness',
                      'Reciprocal', 'Signed_HITS', 'Status_Theory',
                      'Triadic_Balance', 'Triadic_Status', 'Linear_Regression']
    algorithm_list = ['pr', 'bd', 'fg', 'rp', 'sh', 'st', 'tb', 'ts', 'lr']
    # Select Algorithms 
    algorithm_select = [0, 2, 3, 5, 6, 7]
    algorithm_type = [algorithm_type[i] for i in algorithm_select]
    algorithm_list = [algorithm_list[i] for i in algorithm_select]

    rmse = {}
    pcc = {}

    for step, n in enumerate(trainRates):
        GTotal, GTrain = init_graph_oglb(trainEdge, trainWeight, n, testEdge, testWeight, testRate, negEdge, negWeight)
        print('Graph Creation Done')

        algorithm_dict = {}
        for a in algorithm_list:
            if a!='lr':
                algorithm_dict[a] = run_algorithm(a, GTrain)
                rmse[a] = []
                pcc[a] = []
                print(a + ' Done')
        #algorithm_dict['lr'] = Linear_Regression(GTotal, GTrain, algorithm_dict['pr'], algorithm_dict['fg'], algorithm_dict['sh'])

        for key, value in algorithm_dict.items():
            rmse[key].append(predict_weight(value, GTotal, GTrain)[0])
            pcc[key].append(predict_weight(value, GTotal, GTrain)[1])

    rmse_stack = np.vstack(([rmse[each] for each in algorithm_list]))
    pcc_stack = np.vstack(([pcc[each] for each in algorithm_list]))

    df_rmse = pd.DataFrame(rmse_stack, index=algorithm_type, columns=trainRates)
    df_pcc = pd.DataFrame(pcc_stack, index=algorithm_type, columns=trainRates)

    df_rmse.to_csv('./results/leave_10_5_rmse_{}'.format(datasetName))
    df_pcc.to_csv('./results/leave_10_5_pcc_{}'.format(datasetName))

    print('rmse:', df_rmse)
    print('\npcc:', df_pcc)

if __name__ == "__main__":
    main()
