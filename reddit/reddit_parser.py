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
from collections import defaultdict


class RedditParser:
    fields = ['name', 'author', 'subreddit_id', 'subreddit', 'parent_id', 'link_id', 'created_utc', 'score']
    types = {'t1': 'comment', 't2': 'account', 't3': 'link', 't4': 'message', 't5': 'subreddit', 't6': 'award'}
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, from_, to_):
        self.from_year = int(from_[0])
        self.from_month = int(from_[1])
        self.to_year = int(to_[0])
        self.to_month = int(to_[1])
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.datapath = os.path.join(self.path, "data")
        self.files = os.listdir(self.datapath)
        self.files.remove('json')

    def dump_to_json(self):
        for file in self.files:
            sfile = str(file)
            f_month = int(sfile[-6:-4])
            f_year = int(sfile[3:7])
            if self.from_year <= f_year <= self.to_year:
                if self.from_month <= f_month <= self.to_month:
                    dfile = os.path.join(self.datapath, file)
                    jfile = os.path.join(self.datapath, 'json', sfile[:-3] + 'json')
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
    def sampling(data, uniform_p=0.4, min_replies=130, max_replies=16000, max_p=0.08):
        # general uniform sampling
        sample = []
        for tf in data:
            sample.append(np.random.choice(tf, round(uniform_p * len(tf)), replace=False))
        reddits_count = defaultdict(int)
        all = 0
        reddits_to_remove = set()
        final_data = []
        for tf in sample:
            for d in tf:
                reddits_count[d['subreddit']] += 1
                all += 1
        for reddit, count in reddits_count.items():
            r_distr = count / all
            if count < min_replies or count > max_replies or r_distr > max_p:
                reddits_to_remove.add(reddit)
        for tf in sample:
            final_tf = []
            for d in tf:
                if d['subreddit'] not in reddits_to_remove:
                    final_tf.append(d)
            # else:
            #         if d['subreddit'] in reddits_count:
            #             del reddits_count[d['subreddit']]
            final_data.append(final_tf)
        return final_data

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
    def get_week_stats_from_json(date, distribution=False,
                                 uniform_p=0.4, min_replies=130, max_replies=16000, max_p=0.08):
        sampling = True
        sampling_str = '_sample'
        if uniform_p == 1 or min_replies == 1 or max_replies == float('inf') or max_p == 1:
            sampling = False
            sampling_str = ''
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month
        distr = '_distr' if distribution is True else ''
        with open('data/json/' + filename + '.json', 'r') as file, \
                open(os.path.join(RedditParser.path, 'stats',
                                  '{0}_weeks{1}{2}.txt'.format(filename, distr, sampling_str)), 'w+') as fweeks:
            red = [set(), set(), set(), set()]
            acc = [set(), set(), set(), set()]
            name = [set(), set(), set(), set()]
            subs = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
            i = [0, 0, 0, 0]
            wdata = [[], [], [], []]
            for l in file:
                data = json.loads(l)
                if data['author'] != '[deleted]':
                    day = int(data['created_utc'][-2:])
                    week = (day - 1) // 7
                    if week < 4:
                        keep_keys = ['name', 'author', 'subreddit', 'parent_id']
                        data = dict((k, data[k]) for k in keep_keys if k in data)
                        wdata[week].append(data)
                        ########
            # sampling
            wdata = RedditParser.sampling(wdata, uniform_p, min_replies, max_replies)
            for w in range(4):
                for cur_data in wdata[w]:
                    acc[w].add(cur_data['author'])
                    subreddit = cur_data['subreddit']
                    red[w].add(subreddit)
                    name[w].add(cur_data['name'])
                    subs[w][subreddit] += 1
                i[w] = len(wdata[w])

            nrmlz = i if distribution else [1, 1, 1, 1]  # normalizers
            fweeks.write(filename)
            fweeks.write('\n----------------')
            fweeks.write('\nSampling: ' + ", uniform_p=" + str(uniform_p) + ", min_replies=" + str(min_replies)
                         + ", max_replies=" + str(max_replies) + ", max_p=" + str(max_p))
            fweeks.write('\n----------------')
            for w in range(4):
                fweeks.write('\nWeek ' + str(w))
                fweeks.write('\n--------------')
                fweeks.write("\nreddits: " + str(len(red[w])))
                fweeks.write("\naccounts: " + str(len(acc[w])))
                fweeks.write("\nnames: " + str(len(name[w])))
                fweeks.write('\nall posts: ' + str(i[w]) + '\n')
                fweeks.write('\nSubreddits: posts +- from previous week\n')
                sorted_reds = sorted(subs[w].items(), key=lambda k_v: k_v[1], reverse=True)
                for sr in sorted_reds:
                    diff = ''
                    prev = 0
                    if w > 0:
                        if sr[0] in subs[w - 1]:  # check so that it won't create a new key with 0 value
                            prev = subs[w - 1][sr[0]] / nrmlz[w - 1]
                        diff = int(sr[1] - prev) if nrmlz[w] == 1 else sr[1] / nrmlz[w] - prev
                        diff = str(diff) if diff < 0 else '+' + str(diff)
                    cc = sr[1] if nrmlz[w] == 1 else sr[1] / nrmlz[w]
                    fweeks.write('\t' + sr[0] + ": " + str(cc) + ' ' + diff + '\n')

                # NEW and DEAD
                if w > 0:
                    new = set(subs[w].keys()) - set(subs[w - 1].keys())
                    dead = set(subs[w - 1].keys()) - set(subs[w].keys())
                    print('NEW ', "Week ", w, "||   Num: ", len(new))
                    for k, _ in sorted_reds:
                        if k in new:
                            print(k, ": ", subs[w][k], end=',  ')
                    print()
                    print('----')
                    print('DEAD ', "Week ", w, "||   Num: ", len(dead))
                    for k, _ in prev_sorted_reds:
                        if k in dead:
                            print(k, ": ", subs[w - 1][k], end=',  ')
                    print()
                    print()
                prev_sorted_reds = sorted_reds
            fweeks.write('all: ' + str(sum(i)))

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
                    data[d['name']] = {'author': d['author'], 'pid': d['parent_id'], 'subreddit': d['subreddit']}
                    # 'time': d['created_utc']}
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
                    edges.append((attrs1['author'], attrs2['author'], {'subreddit': attrs1['subreddit']}))
                    # 'time': attrs1['time']}))
            G.add_edges_from(edges)
            # deg = nx.degree(G)
            remove = [node for node, degree in G.degree().items() if degree < 2]
            G.remove_nodes_from(remove)

            # Write graph to GML
            gf = "RC" + "_" + str(year) + "-" + str(month) + ".gml"
            nx.write_gml(G, os.path.join('data', 'graphs', gf))

    @staticmethod
    def read_data_from_json(date):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month + '.json'
        wdata = [[], [], [], []]
        with open('data/json/' + filename, 'r') as file:
            for l in file:
                data = json.loads(l)
                if data['author'] != '[deleted]':
                    day = int(data['created_utc'][-2:])
                    week = (day - 1) // 7
                    if week < 4:
                        keep_keys = ['name', 'author', 'subreddit', 'parent_id']
                        data = dict((k, data[k]) for k in keep_keys if k in data)
                        wdata[week].append(data)
        return wdata

    @staticmethod
    def create_graph_files(data, year, month, min_degree=1):
        year = str(year)
        month = str(month) if month >= 10 else '0' + str(month)
        for tfi in range(len(data)):
            G = nx.Graph()
            datadict = {}
            for d in data[tfi]:
                datadict[d['name']] = {'author': d['author'], 'pid': d['parent_id'], 'subreddit': d['subreddit']}
            edges = []
            i = 0
            for attrs1 in datadict.values():
                if i % 20000 == 0:
                    print(i)
                i += 1
                pid = attrs1['pid']
                if pid in datadict.keys():
                    attrs2 = datadict[pid]
                    edges.append((attrs1['author'], attrs2['author'], {'subreddit': attrs1['subreddit']}))
                    # 'time': attrs1['time']}))
            G.add_edges_from(edges)
            remove = [node for node, degree in G.degree().items() if degree < min_degree]
            G.remove_nodes_from(remove)
            # Write graph to GML
            gf = "RC" + "_" + year + "-" + month + "_" + str(tfi) + ".gml"
            nx.write_gml(G, os.path.join('data', 'graphs', gf))

    @staticmethod
    def read_graphs(dates, sampling_p=None, mean_degree_sampling=False):
        Gs = []
        for date in dates:
            year = str(date[0])
            month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
            for tfi in range(4):
                filename = 'RC_' + year + '-' + month + '_' + str(tfi) + '.gml'
                G = nx.read_gml(os.path.join('data', 'graphs', filename))
                G.name = 'RC_' + year + '-' + month + '_' + str(tfi)
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
    def graph_stats(G, file=None):
        info = "Network info:\n" + nx.info(G)
        print(info)
        file.write(info + '\n')
        dens = "Network density: " + str(nx.density(G))
        print(dens)
        file.write(dens + '\n')
        con = "Network connected?: " + str(nx.is_connected(G))
        print(con)
        file.write(con + '\n')
        # if nx.is_connected(G):
        #     diam = "Network diameter: " + str(nx.diameter(G))
        #     print(diam)
        #     f.write(diam)
        avg_cl = 'Average clustering coeff: ' + str(nx.average_clustering(G))
        print(avg_cl)
        file.write(avg_cl + '\n')
        trans = "Triadic closure: " + str(nx.transitivity(G))
        print(trans)
        file.write(trans + '\n')
        pear = 'Degree Pearson corr coeff: ' + str(nx.degree_pearson_correlation_coefficient(G))
        print(pear)
        file.write(pear + '\n')

    @staticmethod
    def get_k_largest_components(G, k=30, min_nodes=3, file=None):
        components = list(nx.connected_component_subgraphs(G))
        comp = 'Number of components: ' + str(len(components))
        print(comp)
        file.write(comp + '\n')
        maxc = 'Largest component nodes: ' + str(len(max(components, key=len)))
        print(maxc)
        file.write(maxc + '\n')
        minc = 'Smallest component nodes: ' + str(len(min(components, key=len)))
        print(minc)
        file.write(minc + '\n')
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


def write_stats():
    Gs = RedditParser.read_graphs([(2010, 9)])

    for G in Gs:
        f = open(os.path.join(RedditParser.path, 'stats', G.name + '_stats.txt'), 'w+')

        print('\n\n--- Main Graph ---')
        f.write('\n\n--- Main Graph ---\n')
        RedditParser.graph_stats(G, f)
        components = RedditParser.get_k_largest_components(G, 20, file=f)

        for c in components:
            print('\n- Component -')
            f.write('\n- Component -\n')
            RedditParser.graph_stats(c, f)
        f.close()


# RedditParser.get_week_stats_from_json((2010, 9), distribution=False)
# data = RedditParser.read_data_from_json((2010, 9))
# data = RedditParser.sampling(data, uniform_p=0.4, min_replies=130, max_replies=16000, max_p=0.08)
# RedditParser.create_graph_files(data, year=2010, month=9)

write_stats()


# write_stats()
# G = RedditParser.read_graphs([(2010, 9)])
# G = RedditParser.community_detection(G)
# dic = defaultdict(int)
# for e in G.edges(data=True):
#     nodes = G.nodes(data=True)
#     n1 = G.nodes(data=True)[e[0]]
#     n2 = G.nodes(data=True)[e[1]]
#     subreddit1 = n1['subreddit']
#     subreddit2 = n2['subreddit']
#     com = e[2]['modularity']
#     dic[com][subreddit1] += 1
#     dic[com][subreddit2] += 1
#
# print(dic)



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
