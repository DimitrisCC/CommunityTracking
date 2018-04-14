import json
import bz2
import os
import datetime


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
    def create_graph_from_json(date):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0' + str(date[1])
        filename = 'RC_' + year + '-' + month + '.json'
        with open('data/json/' + filename, 'r') as file:
            for l in file:
                data = json.loads(l)
                

    @staticmethod
    def get_stats_from_json(date):
        year = str(date[0])
        month = str(date[1]) if date[1] >= 10 else '0'+str(date[1])
        filename = 'RC_' + year + '-' + month + '.json'
        with open('data/json/'+filename, 'r') as file:
            red, acc, name = set(), set(), set()
            i = 0
            for l in file:
                i += 1
                data = json.loads(l)
                red.add(data['subreddit'])
                acc.add(data['author'])
                name.add(data['name'])
            print("\n"+filename)
            print("reddits: ", len(red))
            print("accounts: ", len(acc))
            print("names: ", len(name))
            print('all: ', i)


rp = RedditParser(from_=(2010, 10), to_=(2010, 10))
rp.dump_to_json()
