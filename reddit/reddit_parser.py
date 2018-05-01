import json
import bz2
import os
import datetime
import pandas as pd
from random import random
import networkx as nx
import community
import matplotlib.pyplot as plt
import numpy as np


class RedditParser:
    fields = ['name', 'author', 'subreddit_id', 'subreddit', 'parent_id', 'link_id', 'created_utc', 'score']
    types = {'t1': 'comment', 't2': 'account', 't3': 'link', 't4': 'message', 't5': 'subreddit', 't6': 'award'}

    def __init__(self, from_, to_):
        self.from_year = int(from_[0])
        self.from_month = int(from_[1])
        self.to_year = int(to_[0])
        self.to_month = int(to_[1])
        path_full = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(path_full, "data")
        self.files = os.listdir(self.path)
        self.files.remove('json')

    def dump_to_json(self):
        for file in self.files:
            sfile = str(file)
            f_month = int(sfile[-6:-4])
            f_year = int(sfile[3:7])
            if self.from_year <= f_year <= self.to_year:
                if self.from_month <= f_month <= self.to_month:
                    dfile = os.path.join(self.path, file)
                    jfile = os.path.join(self.path, 'json', sfile[:-3] + 'json')
                    with bz2.open(dfile, 'r') as data, open(jfile, 'w+') as jsondump:
                        print(sfile)
                        for line in data:
                            dline = line.decode("utf-8")
                            ddict = json.loads(dline)
                            sub_dict = {key: ddict.get(key, None) for key in RedditParser.fields}
                            sub_dict['created_utc'] = datetime.datetime.fromtimestamp(int(sub_dict['created_utc'])) \
                                .strftime('%Y-%m-%d')
                            sub_dict['type'] = RedditParser.types[sub_dict['name'][:2]]
                            json.dump(sub_dict, jsondump, ensure_ascii=False)
                            jsondump.write('\n')

    @staticmethod
    def get_stats_from_json(date, sample_p_post=0.5, sample_p_com=0.333):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month + '.json'
        data = []
        with open('data/json/' + filename, 'r') as file:
            for l in file:
                jl = json.loads(l)
                if jl['type'] == 'comment' and random() < sample_p_com:
                    data.append(jl)
                elif not jl['name'].startswith('t1') and random() < sample_p_post:
                    data.append(jl)
                    print(jl['type'])
                elif not jl['name'].startswith('t1'):
                    print(jl['type'])

        df = pd.DataFrame(data)
        red = df.groupby(['type', 'subreddit']).agg(['count'])
        print(red)
        for key, item in red:
            print(red.get_group(key), "\n\n")

    @staticmethod
    def get_stats_from_json_(date):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month + '.json'
        with open('data/json/' + filename, 'r') as file:
            red, acc, name = set(), set(), set()
            i = 0
            for l in file:
                i += 1
                data = json.loads(l)
                red.add(data['subreddit'])
                acc.add(data['author'])
                name.add(data['name'])
            print("\n" + filename)
            print("reddits: ", len(red))
            print("accounts: ", len(acc))
            print("names: ", len(name))
            print('all: ', i)

    @staticmethod
    def create_graph_from_json(date, p=1):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month + '.json'
        with open('data/json/' + filename, 'r') as file:
            G = nx.Graph()
            data = {}
            for l in file:
                if random() < p:
                    d = json.loads(l)
                    if d['author'] == '[deleted]':
                        continue
                    data[d['name']] = {'author': d['author'], 'pid': d['parent_id'], 'subreddit': d['subreddit'],
                                       'time': d['created_utc']}
            print('OKKKKKKKKKKKKKKKKKKKKKKK')
            print("Length: ", len(data))
            edges = []
            i = 0
            for attrs1 in data.values():
                if i % 50000 == 0:
                    print(i)
                i += 1
                pid = attrs1['pid']
                if pid in data.keys():
                    attrs2 = data[pid]
                    edges.append((attrs1['author'], attrs2['author'], {'subreddit': attrs1['subreddit'],
                                                                       'time': attrs1['time']}))
            G.add_edges_from(edges)
            deg = nx.degree(G)
            remove = [node for node, degree in G.degree().items() if degree < 2]
            G.remove_nodes_from(remove)

            # Write graph to GML
            gf = "RC" + "_" + str(year) + "-" + str(month) + ".gml"
            nx.write_gml(G, os.path.join('data', 'graphs', gf))

    @staticmethod
    def read_graphs(dates, sampling_p=None, mean_degree_sampling=False):
        Gs = []
        for date in dates:
            year = str(date[0])
            month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
            filename = 'RC_' + year + '-' + month + '.gml'
            G = nx.read_gml(os.path.join('data', 'graphs', filename))
            G.name = 'RC_' + year + '-' + month
            print("Read")

            # sampling
            if mean_degree_sampling:
                mean_degree = sum(G.degree().values()) / float(len(G))
                G.remove_nodes_from([n for n in G.nodes() if G.degree(n) < mean_degree])
            if sampling_p is not None:
                ch = list(np.random.choice(G.nodes(), (1 - sampling_p) * nx.number_of_nodes(G), replace=False))
                G = G.subgraph(ch)

            Gs.append(G)

        if len(Gs) == 1:
            return Gs[0]
        else:
            return Gs

    @staticmethod
    def graph_stats(G):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(path, 'stats', G.name + '_stats.txt'), 'w+') as f:
            info = "Network info:\n" + nx.info(G)
            print(info)
            f.write(info)
            dens = "Network density: " + str(nx.density(G))
            print(dens)
            f.write(dens)
            con = "Network connected?: " + str(nx.is_connected(G))
            print(con)
            f.write(con)
            # if nx.is_connected(G):
            #     diam = "Network diameter: " + str(nx.diameter(G))
            #     print(diam)
            #     f.write(diam)
            avg_cl = 'Average clustering coeff: ' + str(nx.average_clustering(G))
            print(avg_cl)
            f.write(avg_cl)
            trans = "Triadic closure: " + str(nx.transitivity(G))
            print(trans)
            f.write(trans)
            pear = 'Degree Pearson corr coeff: ' + str(nx.degree_pearson_correlation_coefficient(G))
            print(pear)
            f.write(pear)

    @staticmethod
    def get_k_largest_components(G, k=30, min_nodes=3):
        components = list(nx.connected_component_subgraphs(G))
        path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(path, 'stats', G.name + '_stats.txt'), 'w+') as f:
            comp = 'Number of components: ' + str(len(components))
            print(comp)
            f.write(comp)
            maxc = 'Largest component nodes: ' + str(len(max(components, key=len)))
            print(maxc)
            f.write(maxc)
            minc = 'Smallest component nodes: ' + str(len(min(components, key=len)))
            print(minc)
            f.write(minc)
        kk = min(k, len(components))
        kk_largest = sorted(components, key=len, reverse=True)[:kk]
        to_return = []
        for c in kk_largest:
            if len(c) >= min_nodes:
                to_return.append(c)
        return to_return

    @staticmethod
    def community_detection(G, method='louvain'):
        if method == 'louvain':
            communities = community.best_partition(G)
            nx.set_node_attributes(G, 'modularity', communities)
        return G

    @staticmethod
    def centrality_as_attributes(G):
        deg = nx.degree_centrality(G)
        nx.set_node_attributes(G, 'degree', deg)
        bet = nx.betweenness_centrality(G)
        nx.set_node_attributes(G, 'betweenness', bet)
        eig = nx.eigenvector_centrality(G)
        nx.set_node_attributes(G, 'eigenvector', eig)
        clos = nx.closeness_centrality(G)
        nx.set_node_attributes(G, 'closeness', clos)
        harm = nx.harmonic_centrality(G)
        nx.set_node_attributes(G, 'harmonic', harm)
        return G

    @staticmethod
    def draw(G, partition):
        # drawing
        size = float(len(set(partition.values())))
        pos = nx.spring_layout(G)
        values = [partition.get(node) for node in G.nodes()]

        # subreddits
        r = 0
        reds = nx.get_edge_attributes(G, 'subreddit')
        for red in reds:
            r += 1
            values = [e for e in G.edges(data=True) if G[e[0]][e[1]]['subreddit'] == red]
            nx.draw_spring(G, cmap=plt.get_cmap('jet'), edgelist=values, edge_color=values, node_size=20,
                           with_labels=False)
            if r == 3:
                break

        # for com in set(partition.values()) :
        #  list_nodes = [nodes for nodes in partition.keys()
        #                            if partition[nodes] == com]
        # if len(list_nodes) < 3:
        #   continue
        plt.axis('off')
        # nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color=values, node_size=20, with_labels=False)

        # nx.draw_networkx_edges(G, pos, alpha=0.75)
        plt.show()

        # spring_pos = nx.spring_layout(G)
        # plt.axis('off')
        # nx.draw_networkx(G, pos=spring_pos, with_labels=False, node_size=30)
        # plt.show()


G = RedditParser.read_graphs([(2010, 9)])

path = os.path.dirname(os.path.abspath(__file__))
f = open(os.path.join(path, 'stats', G.name + '_stats.txt'), 'w+')

print('\n\n--- Main Graph ---')
f.write('\n\n--- Main Graph ---')
RedditParser.graph_stats(G)
components = RedditParser.get_k_largest_components(G, 20)

for c in components:
    print('\n- Component -')
    f.write('\n- Component -')
    RedditParser.graph_stats(c)
f.close()

# rp = RedditParser(from_=(2010, 10), to_=(2010, 10))
# rp.dump_to_json()

# RedditParser.get_stats_from_json((2010, 9), sample_p_post=0.05, sample_p_com=0.005)

# #a = {"yolo": "swag", "yolo": "lel", }
# r = nx.random_regular_graph(d=2, n=5)
# lab = {1: "lel", 2: "swag", 3: "e7r", 4: "fr", 0: "grh"}
# nx.set_node_attributes(r, "labelyol", lab)
# G = nx.Graph()
# G.add_nodes_from({"yolo": {"label1": 5, "label2": "kati"},
#                   "swag": {"label1": 32, "label2": "katiallo"},
#                   "lel": {"label1": 6, "label2": "katiakoma"}})
# # G.add_edges_from({'lel':{'swag':{}, 'yolo':{}}})
#
# s = {'yolo': "haha", "swag": "gr"}
# for v, f in s.values():
#     print(v)
#     print(f)
# print()
