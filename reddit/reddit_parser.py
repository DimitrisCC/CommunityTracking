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

fields = ['name', 'author', 'subreddit_id', 'subreddit', 'parent_id', 'link_id', 'created_utc', 'score']
types = {'t1': 'comment', 't2': 'account', 't3': 'link', 't4': 'message', 't5': 'subreddit', 't6': 'award'}
path = os.path.dirname(os.path.abspath(__file__))


def dump_to_json(from_, to_):
    """
    Reads and extracts the reddit data from the bz2 compressed files.
    :param from_    a tuple of (starting year, starting month)
    :param to_    a tuple of (ending year, ending month)
    """
    from_year = int(from_[0])
    from_month = int(from_[1])
    to_year = int(to_[0])
    to_month = int(to_[1])
    datapath = os.path.join(path, "data")
    files = os.listdir(datapath)
    files.remove('json')
    for file in files:
        sfile = str(file)
        f_month = int(sfile[-6:-4])
        f_year = int(sfile[3:7])
        if from_year <= f_year <= to_year:
            if from_month <= f_month <= to_month:
                dfile = os.path.join(datapath, file)
                jfile = os.path.join(datapath, 'json', sfile[:-3] + 'json')
                with bz2.open(dfile, 'r') as data, open(jfile, 'w+') as jsondump:
                    for line in data:
                        dline = line.decode("utf-8")
                        ddict = json.loads(dline)
                        sub_dict = {key: ddict.get(key, None) for key in fields}
                        sub_dict['created_utc'] = datetime.datetime.fromtimestamp(int(sub_dict['created_utc'])) \
                            .strftime('%Y-%m-%d')
                        sub_dict['type'] = types[sub_dict['name'][:2]]
                        json.dump(sub_dict, jsondump, ensure_ascii=False)
                        jsondump.write('\n')


def sampling(data, uniform_p=0.4, min_replies=130, min_replies_week=5, max_replies=16000, max_p=0.08,
             user_min=8, user_threshold=5):
    """
    Performs the sampling and cleaning.
    :param data:                The dict of reddit data
    :param uniform_p:           Probability input of uniform sampling, e.g. 0.4 means throwing away 60% of the data
    :param min_replies:         Minimum number of replies throughout the timeframes. Only subreddits with more than
                                this are kept.
    :param min_replies_week:    Minimum number of replies per timeframe/week. Less than that and the subreddit
                                is considered dead.
    :param max_replies:         The same as min_replies but for too many replies.
    :param max_p:               The same as max_replies but using a probability slice.
    :param user_min:            Users with less replies throughout the timeframes are deleted
    :param user_threshold:      Users with less replies to a subreedit throughout the timeframes, are deleted
                                from the subreddit.
    :return:                    Sampled data as dict, a string with the parameters
    """
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
        for tfi in range(len(sample)):
            final_tf = []
            for d in sample[tfi]:
                if d['subreddit'] not in reddits_to_remove and d['subreddit'] not in reddits_to_remove_week[tfi]:
                    final_tf.append(d)
            final_data.append(final_tf)
        sample = final_data
    sampling_str = "uniform_p=" + str(uniform_p) + ", min_replies=" + str(min_replies) + ", max_replies=" + str(
        max_replies) + ", max_p=" + str(max_p) + ", user_min=" + str(user_min) + ", user_threshold=" + str(
        user_threshold) + ", min_replies_week=" + str(min_replies_week)
    return sample, sampling_str


def get_stats_from_json(data):
    """
    Simple function that prints some pandas DataFrame stats about the subreddits.
    :param data:    reddit data
    """
    df = pd.DataFrame(data)
    red = df.groupby(['type', 'subreddit']).agg(['count'])
    print(red)
    for key, item in red:
        print(red.get_group(key), "\n\n")


def reddit_overlap_percentage(data, fwrite=None):
    overlap = defaultdict(float)
    all = defaultdict(float)
    for tf in data:
        for i1 in range(len(tf)):
            if i1 % 710 == 0:
                print("\t i1 ", i1)
            for i2 in range(i1 + 1, len(tf)):
                red1 = tf[i1]['subreddit']
                red2 = tf[i2]['subreddit']
                all[(red1, red2)] += 1
                if red1 != red2 and tf[i1]['author'] == tf[i2]['author']:
                    overlap[(red1, red2)] += 1
    for key in overlap:
        overlap[key] /= all[key]
    all = None
    overlap = sorted(overlap.items(), key=lambda k_v: k_v[1], reverse=True)
    if fwrite is not None:
        fwrite.write("Overlapping percentage of subreddits in descending order:\n")
        for red, ov in overlap:
            fwrite.write(str(red) + ": " + str(ov) + "%" + "\n")
    return overlap


def get_week_stats_from_data(data, date, distribution=False, sampling=False, sampling_str=None, high_replies=200,
                             low_replies=11):
    """
    0) Extracts stats about the subreddits and the users.
    1) Creates a "RC_<year>-<month>_weeks.txt" file split into timeframes/weeks with cumulative stats and every
    active subreddit in the timeframe with the total number of replies plus the difference from the last timeframe
    if exists.
    2) Creates a "RC_<year>-<month>_users.txt" file containing the activity of each user in all timeframes.
    3) Prints the number and percentage of users with the lowest and highest activity.
    4) Prints subreddits that were born or died throughout the timeframes.
    :param data:            Reddit data as dict
    :param date:            tuple (year, month)
    :param distribution:    If True then creates an additional file with the suffix "_distr" for the weeks stats
                            containing the distributions instead of the raw numbers
    :param sampling:        True if the data are sampled. Adds a "_sample" suffix.
    :param sampling_str:    The sampling string returned by the sampling function.
    :param high_replies:    The min number of replies of a user considered high activity.
    :param low_replies:     Users with less total replies, are considered low activity.
    :return:
    """
    year = str(date[0])
    month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
    filename = 'RC_' + year + '-' + month
    distr = '_distr' if distribution is True else ''
    sample_ = '_sample' if sampling is True else ''
    with open(os.path.join(path, 'stats',
                           '{0}_weeks{1}{2}.txt'.format(filename, distr, sample_)), 'w+') as fweeks, \
            open(os.path.join(path, 'stats',
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
        low = [0 for l in range(low_replies)]
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
                    for l in range(1, low_replies):
                        if replies == l:
                            low[l - 1] += 1
                    if replies > high_replies:
                        high += 1
        for l in range(1, low_replies):
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


def create_graph_from_json(date, min_degree=1):
    """
    Creates a networkx Graph directly from the json files, returns it and writes it in a gml.
    :param date:        date tuple (year, month)
    :param min_degree:  minimum node degree
    :return:            the networkx graph
    """
    year = str(date[0])
    month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
    filename = 'RC_' + year + '-' + month + '.json'
    jsonpath = os.path.join(path, 'data', 'json')
    with open(os.path.join(jsonpath, filename), 'r') as file:
        G = nx.Graph()
        data = {}
        for l in file:
            d = json.loads(l)
            if d['author'] == '[deleted]':
                continue
            data[d['name']] = {'author': d['author'], 'pid': d['parent_id'], 'subreddit': d['subreddit']}
            # 'time': d['created_utc']}
        print('Data dict created')
        print("Length: ", len(data))
        edges = []
        for attrs1 in data.values():
            pid = attrs1['pid']
            if pid in data.keys():
                attrs2 = data[pid]
                edges.append((attrs1['author'], attrs2['author'], {'subreddit': attrs1['subreddit']}))
                # 'time': attrs1['time']}))
        G.add_edges_from(edges)
        # deg = nx.degree(G)
        remove = [node for node, degree in G.degree().items() if degree < min_degree]
        G.remove_nodes_from(remove)

        # Write graph to GML
        gf = "RC" + "_" + str(year) + "-" + str(month) + ".gml"
        nx.write_gml(G, os.path.join(path, 'data', 'graphs', gf))
        return G


def read_data_from_json(date, days=30, sampled=False, days_overlap=0, specific_reddits=None, specific_users=None):
    """
    Reads data from the json files and cleans the [deleted] users.
    :param date:                Date tuple (year, month)
    :param days:                Number of days in timeframe
    :param sampled:             If True, reads the sampled json
    :param days_overlap:        Number of days over timeframe overlap
    :param specific_reddits:    List of specific subreddits you want to read.
    :param specific_users:      List of specific users you want to read.
    :return:    data dict
    """
    year = str(date[0])
    month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
    sampled_str = '_sampled' if sampled else ''
    filename = 'RC_' + year + '-' + month + sampled_str + '.json'
    ddata = [None for i in range(days)]
    wdata = [[], [], [], [], [], [], [], []]
    jsonpath = os.path.join(path, 'data', 'json')
    with open(os.path.join(jsonpath, filename), 'r') as file:
        for l in file:
            data = json.loads(l)
            if data['author'] != '[deleted]':
                if (specific_reddits and data['subreddit'] in specific_reddits) or not specific_reddits:
                    if (specific_users and data['author'] in specific_users) or not specific_users:
                        day = int(data['created_utc'][-2:])
                        if not sampled_str:
                            keep_keys = ['name', 'author', 'subreddit', 'parent_id', 'created_utc']
                            data = dict((k, data[k]) for k in keep_keys if k in data)
                        ddata[day - 1] = data
                        #
                        week = (day - 1) // 7
                        week_day = day - week * 7
                        if days_overlap == 0 and week > 4:
                            continue
                        try:
                            wdata[week].append(data)
                        except IndexError:
                            wdata.append([])
                            wdata[week].append(data)
                        if week_day <= days_overlap and week != 0:
                            wdata[week - 1].append(data)
    wdata = list(filter(None, wdata))
    return wdata


def create_graph_files(data, year, month, min_degree=1):
    """
    Creates networkx Graph from reddit data and write it to gml.
    :param data:        reddit data
    :param year:        year
    :param month:       month
    :param min_degree:  minimum node degree
    :return:    networkx graph
    """
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
        nx.write_gml(G, os.path.join(path, 'data', 'graphs', gf))
        Gs.append(G)
    return Gs


def write_json_from_sampled(data, date, specific_reddits=None):
    """
    Write json file from sampled data.
    :param data:                reddit data
    :param date:                date tuple
    :param specific_reddits:    list of specific subreddits to write
    """
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
    with open(os.path.join(path, 'data', 'json', filename), 'w+') as jsondump:
        for d in selected:
            json.dump(d, jsondump, ensure_ascii=False)
            jsondump.write('\n')


def read_graphs(dates, tfs=4):
    """
    Read gml graph files into networkx graphs
    :param dates:                   date tuples to read
    :return:
    """
    Gs = []
    for date in dates:
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        for tfi in range(tfs):
            filename = 'RC_' + year + '-' + month + '_' + str(tfi) + '.gml'
            G = nx.read_gml(os.path.join(path, 'data', 'graphs', filename))
            print("Read graph ", G.name)
            Gs.append(G)

    if len(Gs) == 1:
        return Gs[0]
    else:
        return Gs


def graph_stats(G, file=None):
    """
    Prints stats about a graph
    :param G:       networkx Graph
    :param file:    the file to write the stats
    :return:
    """
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


def get_k_largest_components(G, k=30, min_nodes=3, file=None):
    """
    Get k largest components of G
    :param G:           networkx graph
    :param k:           number of components
    :param min_nodes:   only components with more than min_nodes number of nodes
    :param file:        file to write
    :return:            list of components
    """
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


def community_detection(G, method='louvain'):
    """
    A simple community detection run
    :param G:       the graph
    :param method:  the method
    :return:        the communities
    """
    if method == 'louvain':
        communities = community.best_partition(G)
        nx.set_node_attributes(G, 'modularity', communities)
    return communities


def centrality_as_attributes(G):
    """
    Add centrality measures as attributes to a graph
    :param G:   the graph
    :return:    the graph with the added attributes
    """
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


def draw(G, partition):
    """
    A messy draw function.
    :param G:           the graph
    :param partition:   a partition of G
    """
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
    """
    Compute and write graphs stats
    :param Gs:                  the graphs
    :param components_also:     if true, write stats for components too
    :param k:                   number of largest components to extract
    """
    if Gs is None:
        Gs = read_graphs([(2010, 9)])

    f = open(os.path.join(path, 'stats', Gs[0].name[:-2] + '_graph_stats.txt'), 'w+')
    for G in Gs:
        print('\n\n--- Main Graph ', G.name[-1], " ---")
        f.write('\n\n--- Main Graph ' + G.name[-1] + " ---\n")
        graph_stats(G, f)

        if components_also:
            components = get_k_largest_components(G, k, file=f)
            for c in components:
                print('\n- Component -')
                f.write('\n- Component -\n')
                graph_stats(c, f)
    f.close()


def choose_experiment(experiment_num):
    # EXPERIMENTS
    R0 = ['craftit', 'WhatWouldYouRather']
    R1 = ['Techno', 'craftit', 'WhatWouldYouRather']
    R2 = ['AskWomen', 'javascript', 'China', 'Jazz', 'craftit', 'tennis', 'Techno', 'math', 'cars']
    R3 = ['Entlantis', 'WhatWouldYouRather', 'TrueBlood', 'Assistance', 'BodyAcceptance', 'announcements',
          'RedditCarpool', 'PlacetoStay', 'craftit', 'reachclan', 'RedditvFCC', 'playitforward', 'Techno', 'tennis',
          'Coffee', 'women', 'ipad']
    R4 = ['dogs', 'history', 'rpg']
    R5 = ['AMerrickanGirl', 'Blue Rock', 'matts2', 'ChaosMotor', 'RobotBuddha', 'LGBTerrific', 'anutensil',
          'insomniac84', 'enkiam', 'SargonOfAkkad', 'adaminc', 'HeathenCyclist', 'dancer101', 'Issachar',
          'JohnStrangerGalt', 'ieattime20', 'Sommiel', 'drunkentune', 'hotsexgary']
    R6 = ['NaziHunting', 'Khazar_Pride', 'announcements', 'ideasfortheadmins', 'Conservative', 'Mommit', 'TrueBlood',
          'conspiratard', 'houston', 'UniversityOfHouston', '911truth', 'climateskeptics', 'DebateAnAtheist',
          'collapse', 'AskWomen', 'Denver', 'happy', 'Assistance', 'ToolBand', 'software']

    experiments = [(R0, '_R0'), (R1, '_R1'), (R2, '_R2'), (R3, '_R3'), (R4, '_R4'), (R5, '_R5'), (R6, '_R6')]
    return experiments[experiment_num]


EXPERIMENT = 1

if __name__ == "__main__":
    import sys

    # if sys.argv is not None or sys.argv != []:
    #     reds = sys.argv[0]
    # else:
    #     reds, _ = choose_experiment(EXPERIMENT)
    # data = read_data_from_json((2010, 9), sampled=True)
    # # data, sampling_str = sampling(data, uniform_p=0.7, min_replies=130, max_replies=16000, max_p=0.1,
    # #                               user_min=8, user_threshold=5)
    # f = open(os.path.join(path, 'stats', 'overlap.txt'), 'w+')
    # overlap = reddit_overlap_percentage(data)
    # for r, ov in overlap:
    #     print(r, ": ", ov, "%")
    #     f.write(str(r) + ": " + str(ov) + "%")
    # # write_json_from_sampled(data, date=(2010, 9))
    # # data = read_data_from_json((2010, 9), sampled=True)
    # # get_week_stats_from_data(data, (2010, 9), sampling=True, sampling_str=sampling_str)
    # # Gs = create_graph_files(data, year=2010, month=9, min_degree=1)
    # # write_graph_stats(Gs)

    data = read_data_from_json((2010, 10), days=31, sampled=False)
    data, sampling_str = sampling(data, uniform_p=0.7, min_replies=130, max_replies=16000, max_p=0.1,
                                  user_min=8, user_threshold=5)
    write_json_from_sampled(data, date=(2010, 10))
