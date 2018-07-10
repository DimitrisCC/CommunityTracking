from __future__ import division
from muturank import Muturank_new
from synthetic import SyntheticDataConverter
from metrics import evaluate
# from dblp import dblp_loader
import networkx as nx
from itertools import combinations_with_replacement
import random
from tensor import TensorFact
import pickle
from collections import OrderedDict
import time
import json
from tabulate import tabulate
import pprint
from ged_lib import Tracker
from ged_lib import preprocessing


class Data(object):
    def __init__(self, comms, graphs, timeFrames, number_of_dynamic_communities, dynamic_truth=[]):
        self.comms = comms
        self.graphs = graphs
        self.timeFrames = timeFrames
        self.number_of_dynamic_communities = number_of_dynamic_communities
        self.dynamic_truth = dynamic_truth


def object_decoder(obj, num):
    if 'type' in obj[num] and obj[num]['type'] == 'hand':
        edges = {int(tf): [(edge[0], edge[1]) for edge in edges] for tf, edges in obj[num]['edges'].items()}
        graphs = {}
        for i, edges in edges.items():
            graphs[i] = nx.Graph(edges)
        comms = {int(tf): {int(id): com for id, com in coms.items()} for tf, coms in obj[num]['comms'].items()}
        dynamic_coms = {int(id): [str(node) for node in com] for id, com in obj[num]['dynamic_truth'].items()}
        return Data(comms, graphs, len(graphs), len(dynamic_coms), dynamic_coms)
    return obj


def run_experiments(data, ground_truth, network_num):
    times = []
    all_res = []
    # Timerank with one connection - default q
    start_time = time.time()
    mutu4 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
                         clusters=len(ground_truth), default_q=True)
    all_res.append(evaluate.get_results(ground_truth, mutu4.dynamic_coms, "Timerank-STC-Uni", mutu4.tfs,
                                        eval="dynamic", duration=mutu4.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu4.dynamic_coms, "Timerank-STC-Uni", mutu4.tfs,
                                        eval="sets", duration=mutu4.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu4.dynamic_coms, "Timerank-STC-Uni", mutu4.tfs,
                                        eval="per_tf", duration=mutu4.duration))
    duration = time.time() - start_time
    print("Timerank with one connection - default q: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    # Timerank with all connections - default q
    start_time = time.time()
    mutu5 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                         clusters=len(ground_truth), default_q=True)
    all_res.append(evaluate.get_results(ground_truth, mutu5.dynamic_coms, "Timerank-AOC-Uni", mutu5.tfs,
                                        eval="dynamic", duration=mutu5.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu5.dynamic_coms, "Timerank-AOC-Uni", mutu5.tfs,
                                        eval="sets", duration=mutu5.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu5.dynamic_coms, "Timerank-AOC-Uni", mutu5.tfs,
                                        eval="per_tf", duration=mutu5.duration))
    duration = time.time() - start_time
    print("Timerank with all connection - default q: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    # Timerank with next connection - default q
    start_time = time.time()
    mutu6 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='next',
                         clusters=len(ground_truth), default_q=True)
    all_res.append(evaluate.get_results(ground_truth, mutu6.dynamic_coms, "Timerank-NOC-Uni"
                                        , mutu6.tfs, eval="dynamic", duration=mutu6.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu6.dynamic_coms, "Timerank-NOC-Uni",
                                        mutu6.tfs, eval="sets", duration=mutu6.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu6.dynamic_coms, "Timerank-NOC-Uni",
                                        mutu6.tfs, eval="per_tf", duration=mutu6.duration))
    duration = time.time() - start_time
    print("Timerank with next connection - default q: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    # Run Timerank - One connection
    start_time = time.time()
    mutu1 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
                         clusters=len(ground_truth), default_q=False)
    all_res.append(evaluate.get_results(ground_truth, mutu1.dynamic_coms, "Timerank-STC", mutu1.tfs,
                                        eval="dynamic", duration=mutu1.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu1.dynamic_coms, "Timerank-STC", mutu1.tfs,
                                        eval="sets", duration=mutu1.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu1.dynamic_coms, "Timerank-STC", mutu1.tfs,
                                        eval="per_tf", duration=mutu1.duration))
    duration = time.time() - start_time
    print("Timerank with one connection: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    muturank_res = OrderedDict()
    muturank_res["tf/node"] = ['t' + str(tf) for tf in mutu1.tfs_list]
    for i, node in enumerate(mutu1.node_ids):
        muturank_res[node] = [mutu1.p_new[tf * len(mutu1.node_ids) + i] for tf in range(mutu1.tfs)]
    f = open('results_hand.txt', 'a')
    f.write("ONE CONNECTION\n")
    f.write(tabulate(muturank_res, headers="keys", tablefmt="grid") + "\n")
    f.write(tabulate(zip(['t' + str(tf) for tf in mutu1.tfs_list], mutu1.q_new), headers="keys",
                     tablefmt="grid") + "\n")
    f.close()

    # Timerank with all connections
    start_time = time.time()
    mutu2 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                         clusters=len(ground_truth), default_q=False)
    all_res.append(evaluate.get_results(ground_truth, mutu2.dynamic_coms, "Timerank-AOC", mutu2.tfs,
                                        eval="dynamic", duration=mutu2.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu2.dynamic_coms, "Timerank-AOC", mutu2.tfs,
                                        eval="sets", duration=mutu2.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu2.dynamic_coms, "Timerank-AOC", mutu2.tfs,
                                        eval="per_tf", duration=mutu2.duration))
    duration = time.time() - start_time
    print("Timerank with all connection: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    muturank_res = OrderedDict()
    muturank_res["tf/node"] = ['t' + str(tf) for tf in mutu2.tfs_list]
    for i, node in enumerate(mutu2.node_ids):
        muturank_res[node] = [mutu2.p_new[tf * len(mutu2.node_ids) + i] for tf in range(mutu2.tfs)]
    f = open('results_hand.txt', 'a')
    f.write("ALL CONNECTIONS\n")
    f.write(tabulate(muturank_res, headers="keys", tablefmt="grid") + "\n")
    f.write(tabulate(zip(['t' + str(tf) for tf in mutu2.tfs_list], mutu2.q_new), headers="keys",
                     tablefmt="grid") + "\n")
    f.close()

    # Timerank with next connection
    start_time = time.time()
    mutu3 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='next',
                         clusters=len(ground_truth), default_q=False)
    all_res.append(evaluate.get_results(ground_truth, mutu3.dynamic_coms, "Timerank-NOC", mutu3.tfs,
                                        eval="dynamic", duration=mutu3.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu3.dynamic_coms, "Timerank-NOC", mutu3.tfs,
                                        eval="sets", duration=mutu3.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu3.dynamic_coms, "Timerank-NOC", mutu3.tfs,
                                        eval="per_tf", duration=mutu3.duration))
    duration = time.time() - start_time
    print("Timerank with next connection: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    muturank_res = OrderedDict()
    muturank_res["tf/node"] = ['t' + str(tf) for tf in mutu3.tfs_list]
    for i, node in enumerate(mutu3.node_ids):
        muturank_res[node] = [mutu3.p_new[tf * len(mutu3.node_ids) + i] for tf in range(mutu3.tfs)]
    f = open('results_hand.txt', 'a')
    f.write("NEXT CONNECTION\n")
    f.write(tabulate(muturank_res, headers="keys", tablefmt="grid") + "\n")
    f.write(tabulate(zip(['t' + str(tf) for tf in mutu3.tfs_list], mutu3.q_new), headers="keys",
                     tablefmt="grid") + "\n")
    f.write("GROUND TRUTH\n")
    pprint.pprint(ground_truth, stream=f, width=150)
    f.write("ONE CONNECTION\n")
    pprint.pprint(mutu1.dynamic_coms, stream=f, width=150)
    f.write("ALL CONNECTIONS\n")
    pprint.pprint(mutu2.dynamic_coms, stream=f, width=150)
    f.write("NEXT CONNECTION\n")
    pprint.pprint(mutu3.dynamic_coms, stream=f, width=150)
    f.close()

    # NNTF
    start_time = time.time()
    fact = TensorFact(data.graphs, num_of_coms=len(ground_truth), threshold=1e-4, seeds=1, overlap=False)
    fact_dur = time.time() - start_time
    fact_dur = "%d:%d" % (fact_dur // 60, fact_dur % 60)
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF", mutu6.tfs, eval="dynamic",
                                        duration=fact_dur))
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF", mutu6.tfs, eval="sets",
                                        duration=fact_dur))
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF", mutu6.tfs, eval="per_tf",
                                        duration=fact_dur))
    duration = time.time() - start_time
    print("NNTF: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    with open('results_hand.txt', 'a') as f:
        f.write("NNTF\n")
        f.write("Error: " + str(fact.error) + "Seed: " + str(fact.best_seed) + "\n")
        f.write("A\n")
        pprint.pprint(fact.A, stream=f, width=150)
        f.write("B\n")
        pprint.pprint(fact.B, stream=f, width=150)
        f.write("C\n")
        pprint.pprint(fact.C, stream=f, width=150)
        pprint.pprint(fact.dynamic_coms, stream=f, width=150)

    # NNTF-Timerank tensor
    new_graphs = {}
    for i, A in mutu1.a.items():
        new_graphs[i] = nx.from_scipy_sparse_matrix(A)
    start_time = time.time()
    fact2 = TensorFact(new_graphs, num_of_coms=len(ground_truth), threshold=1e-4, seeds=1, overlap=False,
                       original_graphs=data.graphs)
    fact_dur = time.time() - start_time
    fact_dur = "%d:%d" % (fact_dur // 60, fact_dur % 60)
    all_res.append(evaluate.get_results(ground_truth, fact2.dynamic_coms, "NNTF-Timerank tensor", mutu6.tfs,
                                        eval="dynamic", duration=fact_dur))
    all_res.append(evaluate.get_results(ground_truth, fact2.dynamic_coms, "NNTF-Timerank tensor", mutu6.tfs,
                                        eval="sets", duration=fact_dur))
    all_res.append(evaluate.get_results(ground_truth, fact2.dynamic_coms, "NNTF-Timerank tensor", mutu6.tfs,
                                        eval="per_tf", duration=fact_dur))
    duration = time.time() - start_time
    print("NNTF-Timerank tensor: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    with open('results_hand.txt', 'a') as f:
        f.write("NNTF\n")
        f.write("Error: " + str(fact2.error) + "Seed: " + str(fact2.best_seed) + "\n")
        f.write("A\n")
        pprint.pprint(fact2.A, stream=f, width=150)
        f.write("B\n")
        pprint.pprint(fact2.B, stream=f, width=150)
        f.write("C\n")
        pprint.pprint(fact2.C, stream=f, width=150)
        pprint.pprint(fact2.dynamic_coms, stream=f, width=150)

    # NNTF-Overlap
    start_time = time.time()
    fact = TensorFact(data.graphs, num_of_coms=len(ground_truth), threshold=1e-4, seeds=1, overlap=True)
    fact_dur = time.time() - start_time
    fact_dur = "%d:%d" % (fact_dur // 60, fact_dur % 60)
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF-Overlap", mutu6.tfs, eval="dynamic",
                                        duration=fact_dur))
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF-Overlap", mutu6.tfs, eval="sets",
                                        duration=fact_dur))
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF-Overlap", mutu6.tfs, eval="per_tf",
                                        duration=fact_dur))
    duration = time.time() - start_time
    print("NNTF-Overlap: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    with open('results_hand.txt', 'a') as f:
        f.write("NNTF-Overlap\n")
        f.write("Error: " + str(fact.error) + "Seed: " + str(fact.best_seed) + "\n")
        f.write("A\n")
        pprint.pprint(fact.A, stream=f, width=150)
        f.write("B\n")
        pprint.pprint(fact.B, stream=f, width=150)
        f.write("C\n")
        pprint.pprint(fact.C, stream=f, width=150)
        pprint.pprint(fact.dynamic_coms, stream=f, width=150)

    # GED
    import sys
    sys.path.insert(0, '../GED/')
    start_time = time.time()
    from ged import GedWrite, ReadGEDResults
    ged_data = GedWrite(data)
    graphs = preprocessing.getGraphs(ged_data.fileName)
    tracker = Tracker.Tracker(graphs)
    tracker.compare_communities()
    if not os.path.exists('results'):
        os.makedirs('results')
    outfile = os.path.join('results', 'GED-events-handdrawn' + str(network_num) + '.csv')

    with open(outfile, 'w+') as f:
        for hypergraph in tracker.hypergraphs:
            hypergraph.calculateEvents(f)

    ged = ReadGEDResults.ReadGEDResults(file_coms=ged_data.fileName, file_output=outfile)
    ged_dur = time.time() - start_time
    ged_dur = "%d:%d" % (ged_dur // 60, ged_dur % 60)
    with open('results_hand.txt', 'a') as f:
        f.write("GED\n")
        pprint.pprint(ged.dynamic_coms, stream=f, width=150)
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED-T", mutu6.tfs, eval="dynamic",
                                        duration=ged_dur))
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED-T", mutu6.tfs, eval="sets",
                                        duration=ged_dur))
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED-T", mutu6.tfs, eval="per_tf",
                                        duration=ged_dur))
    duration = time.time() - start_time
    print("GED-T: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    # GED with timerank communities
    # GED
    import sys
    sys.path.insert(0, '../GED/')
    start_time = time.time()
    from ged import GedWrite, ReadGEDResults
    ged_data = GedWrite(Data(mutu1.comms, data.graphs, len(graphs), len(mutu1.dynamic_coms), mutu1.dynamic_coms))
    graphs = preprocessing.getGraphs(ged_data.fileName)
    tracker = Tracker.Tracker(graphs)
    tracker.compare_communities()
    outfile = './results/GED-events-handdrawn-' + str(network_num) + '.csv'

    with open(outfile, 'w') as f:
        for hypergraph in tracker.hypergraphs:
            hypergraph.calculateEvents(f)

    ged = ReadGEDResults.ReadGEDResults(file_coms=ged_data.fileName, file_output=outfile)
    ged_dur = time.time() - start_time
    ged_dur = "%d:%d" % (ged_dur // 60, ged_dur % 60)
    with open('results_hand.txt', 'a') as f:
        f.write("GED\n")
        pprint.pprint(ged.dynamic_coms, stream=f, width=150)
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED - with Timerank comms", mutu6.tfs,
                                        eval="dynamic", duration=ged_dur))
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED - with Timerank comms", mutu6.tfs,
                                        eval="sets", duration=ged_dur))
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED - with Timerank comms", mutu6.tfs,
                                        eval="per_tf", duration=ged_dur))
    duration = time.time() - start_time
    print("GED - with Timerank comms: TIME: %d min, %d sec" % (duration // 60, duration % 60))
    times.append(duration)

    print("TOTAL TIME: %d min, %d sec" % (sum(times) // 60, sum(times) % 60))
    return all_res


def create_ground_truth(communities, number_of_dynamic_communities):
    ground_truth = {i: [] for i in range(number_of_dynamic_communities)}
    for tf, coms in communities.items():
        for i, com in coms.items():
            for node in com:
                ground_truth[i].append(str(node) + "-t" + str(tf))
    return ground_truth


if __name__ == "__main__":

    # ---------------------------------
    # dblp = dblp_loader("data/dblp/my_dblp_data.json", start_year=2000, end_year=2004, coms='comp')
    # number_of_dynamic_communities = len(dblp.dynamic_coms)
    # data = Data(dblp.communities, dblp.graphs, len(dblp.graphs), len(dblp.dynamic_coms))
    # ground_truth = dblp.dynamic_coms
    # ---------------------------------
    import os

    path_full = os.path.dirname(os.path.abspath(__file__))
    path_full = os.path.join(path_full, "data", "hand-drawn-data.json")

    with open(path_full, mode='r') as fp:
        hand_drawn = json.load(fp)
    f = open('results_hand.txt', 'a')
    f.write('')
    f.close()
    for i in range(len(hand_drawn)):
        # for i in [2]:
        data = object_decoder(hand_drawn, i)
        # from plot import PlotGraphs
        # PlotGraphs(data.graphs, len(data.graphs), 'hand-written'+str(i), 500)
        f = open('results_hand.txt', 'a')
        f.write("\n" + "-" * 80 + "NETWORK #" + str(hand_drawn[i]['id']) + "-" * 80 + "\n")
        f.close()
        print(hand_drawn[i]['id'])
        all_res = run_experiments(data, data.dynamic_truth, hand_drawn[i]['id'])
        results = OrderedDict()
        results["Method"] = []
        results['Eval'] = []
        results['NMI'] = []
        results['Omega'] = []
        results['Bcubed-Precision'] = []
        results['Bcubed-Recall'] = []
        results['Bcubed-F1'] = []
        results['Duration'] = []
        for res in all_res:
            for k, v in res.items():
                results[k].extend(v)
        f = open('results_hand.txt', 'a')
        f.write(tabulate(results, headers="keys", tablefmt="grid") + "\n")
        import pandas as pd

        df = pd.DataFrame.from_dict(results)
        del df["Duration"]
        f.write("\\begin{table}[h!] \n\centering \n\\begin{tabular}{ |p{4cm}||p{2cm}|p{3cm}|p{2cm}|p{2cm}|p{2cm}|} "
                "\n"
                "\hline \n\multicolumn{6}{|c|}{Evaluation comparison} \\\\\n\hline\n Method& NMI & Omega Index & "
                "BCubed Precision & BCubed Recall & BCubed F\\\\\n\hline\n")
        for index, row in df.iterrows():
            if row["Eval"] == "dynamic":
                del row["Eval"]
                f.write(str(row[0]))
                for item in row[1:]:
                    f.write(" & " + str(item))
                f.write(str("\\\\") + "\n")
        f.write("\hline\n\end{tabular}\n\caption{Comparison of different frameworks on Hand-drawn Network \\# "
                "" + str(hand_drawn[i]['id']) +
                " illustrated in \\ref{fig: network" + str(hand_drawn[i]['id']) + "} }"
                                                                                  "\n\label{table:results-network" + str(
            hand_drawn[i]['id']) + "}\n\end{table}\n")
        f.close()
