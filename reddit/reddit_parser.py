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
    def sampling(data, uniform_p=0.4, min_replies=130, min_replies_week=5, max_replies=16000, max_p=0.08,
                 user_min=8, user_threshold=5):
        if uniform_p == 1 and min_replies <= 1 and max_replies == float(
                'inf') and max_p == 1 and min_replies_week <= 1 and user_min <= 1 and user_threshold <= 1:
            return data
        # general uniform sampling
        sample = []
        if uniform_p != 1:
            for tf in data:
                sample.append(np.random.choice(tf, round(uniform_p * len(tf)), replace=False))
        else:
            sample = data

        # users sampling
        if user_min > 1 or user_threshold > 1:
            users = defaultdict(lambda: defaultdict(int))
            for tf in data:
                for d in tf:
                    users[d['author']][d['subreddit']] += 1
                    users[d['author']]['all'] += 1
            users_to_remove = set()
            users_reds_to_remove = defaultdict(list)
            for user in users:
                if users[user]['all'] < user_min:  # if user is not active enough
                    users_to_remove.add(user)
                else:
                    for red in users[user]:
                        if users[user][red] < user_threshold:  # if the user contributed far too little to some reddits
                            users_reds_to_remove[user].append(red)
            final_data = []
            for tf in data:
                final_tf = []
                for d in tf:
                    duser = d['author']
                    if duser not in users_to_remove and d['subreddit'] not in users_reds_to_remove[duser]:
                        final_tf.append(d)
                final_data.append(final_tf)
            sample = final_data

        # reddit sampling
        if min_replies > 1 or max_replies != float('inf') or max_p < 1 or min_replies_week > 1:
            reddits_count = defaultdict(lambda: defaultdict(int))
            all = 0
            reddits_to_remove = set()
            reddits_to_remove_week = [set() for tf in sample]
            final_data = []
            for tfi in range(len(sample)):
                for d in sample[tfi]:
                    reddits_count[d['subreddit']][tfi] += 1
                    reddits_count[d['subreddit']]['all'] += 1
                    all += 1
            for reddit, rtfs in reddits_count.items():
                count = rtfs['all']
                r_distr = count / all
                if count < min_replies or count > max_replies or r_distr > max_p:
                    reddits_to_remove.add(reddit)
                else:
                    for tfi in range(len(sample)):
                        if rtfs[tfi] < min_replies_week:  # those subreddits are considered inactive/dead
                            reddits_to_remove_week[tfi].add(reddit)
            for tf in sample:
                final_tf = []
                for d in tf:
                    if d['subreddit'] not in reddits_to_remove:
                        final_tf.append(d)
                final_data.append(final_tf)
            sample = final_data
        sampling_str = "uniform_p=" + str(uniform_p) + ", min_replies=" + str(min_replies) + ", max_replies=" + str(
            max_replies) + ", max_p=" + str(max_p) + ", user_min=" + str(user_min) + ", user_threshold=" + str(
            user_threshold) + ", min_replies_week=" + str(min_replies_week)
        return sample, sampling_str

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
    def get_week_stats_from_data(data, date, distribution=False, sampling=False, sampling_str=None):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month
        distr = '_distr' if distribution is True else ''
        sample_ = '_sample' if sampling is True else ''
        with open(os.path.join(RedditParser.path, 'stats',
                               '{0}_weeks{1}{2}.txt'.format(filename, distr, sample_)), 'w+') as fweeks, \
                open(os.path.join(RedditParser.path, 'stats',
                                  '{0}_users{1}.txt'.format(filename, sample_)), 'w+') as fusers:
            reddits = [set(), set(), set(), set()]
            acc = [set(), set(), set(), set()]
            name = [set(), set(), set(), set()]
            accounts = set()
            subs = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
            users = defaultdict(lambda: defaultdict(int))
            i = [0, 0, 0, 0]

            for w in range(4):
                for cur_data in data[w]:
                    acc[w].add(cur_data['author'])
                    subreddit = cur_data['subreddit']
                    reddits[w].add(subreddit)
                    name[w].add(cur_data['name'])
                    subs[w][subreddit] += 1
                    users[cur_data['author']][cur_data['subreddit']] += 1
                    users[cur_data['author']]['all'] += 1
                    accounts.add(cur_data['author'])
                i[w] = len(data[w])

            # USERS
            _low = 11
            low = [0 for l in range(_low)]
            high = 0
            fusers.write(filename)
            fusers.write('\n----------------')
            fusers.write('\nSampling: ' + sampling_str)
            fusers.write('\n----------------')
            sorted_u = sorted(users.items(), key=lambda k_v: k_v[1]['all'], reverse=True)
            fusers.write('\nWHOLE MONTH')
            for user in sorted_u:
                user = user[0]
                fusers.write("\n" + user)
                sorted_r = sorted(users[user].items(), key=lambda k_v: k_v[1], reverse=True)
                for red in sorted_r:
                    red = red[0]
                    replies = users[user][red]
                    fusers.write("\n    " + red + ": " + str(replies))
                    if red == 'all':
                        for l in range(1, _low):
                            if replies == l:
                                low[l - 1] += 1
                        if replies > 200:
                            high += 1
            for l in range(1, _low):
                print("Low ", l, " replies: ", low[l - 1], "    Percentage:", low[l - 1] / len(accounts))
            print("Low all: ", sum(low), "    Percentage:", sum(low) / len(accounts))
            print("High: ", high, "    Percentage:", high / len(accounts))
            print("ALL: ", len(accounts))
            #

            nrmlz = i if distribution else [1, 1, 1, 1]  # normalizers
            fweeks.write(filename)
            fweeks.write('\n----------------')
            fweeks.write('\nSampling: ' + sampling_str)
            fweeks.write('\n----------------')
            for w in range(4):
                fweeks.write('\nWeek ' + str(w))
                fweeks.write('\n--------------')
                fweeks.write("\nreddits: " + str(len(reddits[w])))
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
    def read_data_from_json(date, sampled=False, specific_reddits=None):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        sampled_str = '_sampled' if sampled else ''
        filename = 'RC_' + year + '-' + month + sampled_str + '.json'
        wdata = [[], [], [], []]
        with open('data/json/' + filename, 'r') as file:
            for l in file:
                data = json.loads(l)
                if data['author'] != '[deleted]':
                    if (specific_reddits and data['subreddit'] in specific_reddits) or not specific_reddits:
                        day = int(data['created_utc'][-2:])
                        week = (day - 1) // 7
                        if week < 4:
                            if not sampled_str:
                                keep_keys = ['name', 'author', 'subreddit', 'parent_id', 'created_utc']
                                data = dict((k, data[k]) for k in keep_keys if k in data)
                            wdata[week].append(data)
        return wdata

    @staticmethod
    def create_graph_files(data, year, month, min_degree=2):
        year = str(year)
        month = str(month) if month >= 10 else '0' + str(month)
        Gs = []
        for tfi in range(len(data)):
            G = nx.Graph()
            datadict = {}
            for d in data[tfi]:
                datadict[d['name']] = {'author': d['author'], 'pid': d['parent_id'], 'subreddit': d['subreddit']}
            edges = []
            i = 0
            for attrs1 in datadict.values():
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
            G.name = "RC" + "_" + year + "-" + month + "_" + str(tfi)
            nx.write_gml(G, os.path.join('data', 'graphs', gf))
            Gs.append(G)
        return Gs

    @staticmethod
    def get_communities_from_graph(G, contribution_threshold=0):
        """
        :param G: the graph of a timeframe
        :param contribution_threshold: the percentage of user's replies needed to be assigned to a community
                0 means maximum overlapping, 1 or anything larger than the largest contribution of a user to
                a community means no overlapping
        :return: dict of communities
        """
        pass

    @staticmethod
    def write_json_from_sampled(data, date, specific_reddits=None):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month + '_sampled.json'
        selected = []
        for week_data in data:
            for d in week_data:
                if specific_reddits:
                    if d['subreddit'] in specific_reddits:
                        selected.append(d)
                else:
                    selected.append(d)
        with open(os.path.join(RedditParser.path, 'data', 'json', filename), 'w+') as jsondump:
            for d in selected:
                json.dump(d, jsondump, ensure_ascii=False)
                jsondump.write('\n')

    @staticmethod
    def read_graphs(dates, sampling_p=None, mean_degree_sampling=False):
        Gs = []
        for date in dates:
            year = str(date[0])
            month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
            for tfi in range(4):
                filename = 'RC_' + year + '-' + month + '_' + str(tfi) + '.gml'
                G = nx.read_gml(os.path.join(RedditParser.path, 'data', 'graphs', filename))
                print("Read graph ", G.name)

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


def write_graph_stats(Gs, components_also=False, k=20):
    if Gs is None:
        Gs = RedditParser.read_graphs([(2010, 9)])

    f = open(os.path.join(RedditParser.path, 'stats', Gs[0].name[:-2] + '_graph_stats.txt'), 'w+')
    for G in Gs:
        print('\n\n--- Main Graph ', G.name[-1], " ---")
        f.write('\n\n--- Main Graph ' + G.name[-1] + " ---\n")
        RedditParser.graph_stats(G, f)

        if components_also:
            components = RedditParser.get_k_largest_components(G, k, file=f)
            for c in components:
                print('\n- Component -')
                f.write('\n- Component -\n')
                RedditParser.graph_stats(c, f)
    f.close()


if __name__ == "__main__":
    data = RedditParser.read_data_from_json((2010, 9), sampled=True,
                                            specific_reddits=['AskWomen', 'javascript', 'China', 'Jazz', 'craftit',
                                                              'tennis', 'Techno', 'math', 'cars'])
    # specific_reddits=['Techno', 'craftit', 'WhatWouldYouRather', 'trackers',
    #                   'depression'])
    # data, sampling_str = RedditParser.sampling(data, uniform_p=0.7, min_replies=130, max_replies=16000, max_p=0.1,
    #                                            user_min=8, user_threshold=5)
    # RedditParser.write_json_from_sampled(data, date=(2010, 9))
    # data = RedditParser.read_data_from_json((2010, 9), sampled=True)
    # RedditParser.get_week_stats_from_data(data, (2010, 9), sampling=True, sampling_str=sampling_str)
    Gs = RedditParser.create_graph_files(data, year=2010, month=9, min_degree=1)
    # write_graph_stats(Gs)

    # year = "2010"
    # month = "09"
    # tfi = 1
    # filename = 'RC_' + year + '-' + month + '_' + str(tfi) + '.gml'
    # G = nx.read_gml(os.path.join('data', 'graphs', filename))
    # G.name = 'RC_' + year + '-' + month + '_' + str(tfi)
    # nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_size=20, with_labels=False)
    # plt.show()

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
