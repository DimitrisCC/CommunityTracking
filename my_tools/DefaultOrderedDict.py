from collections import OrderedDict, Callable, defaultdict


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, sort_key=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory
        self.sort_key = sort_key

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))

    def __lt__(self, other):
        if self.sort_key in self and self.sort_key in other:
            return self[self.sort_key] < other[self.sort_key]
        else:
            return False


if __name__ == "__main__":
    a = DefaultOrderedDict(lambda: DefaultOrderedDict(int, 'dnd'))
    a = defaultdict(lambda: DefaultOrderedDict(int, 'dnd'))
    b = DefaultOrderedDict(int)

    a['adam']['dnd'] += 5
    a['adam']['tlt'] += 1
    a['jay']['dnd'] += 25
    a['gar']['dnd'] += 8
    a['gar']['lel'] += 2
    a['gar']['rel'] += 85
    a['gar']['avl'] += 9
    a['gar']['red'] += -2

    b['dnd'] += 8
    b['lel'] += 2
    b['rel'] += 85
    b['avl'] += 9
    b['red'] += -2

    for u in b:
        print(u, ": ", b[u])

        # sorted_reds = sorted(a.items(), key=lambda k_v: k_v[1]['dnd'], reverse=True)
        # print(sorted_reds)
        # for user in sorted_reds:
        #     print(user[0])
        #     for red in a[user[0]]:
        #         print("   ", red, ": ", a[user[0]][red])
