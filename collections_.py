from collections import Counter


class Counter_(Counter):
    def __add__(self, other):
        '''Add counts from two counters.

        >>> Counter_('abbb') + Counter_('bcc')
        Counter({'b': 4, 'c': 2, 'a': 1})

        '''
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter_()
        for elem, count in self.items():
            newcount = count + other[elem]
            if all([newcount]) > 0:
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and all([count]) > 0:
                result[elem] = count
        return result
