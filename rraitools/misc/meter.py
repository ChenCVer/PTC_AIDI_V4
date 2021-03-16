class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.__val = 0
        self.__avg_val = 0
        self.__sum_val = 0
        self.__count_val = 0
        self.__max_val = -1e12
        self.__min_val = 1e12

    def update(self, val, n=1):
        self.__val = val
        self.__sum_val += val * n
        self.__count_val += n
        self.__avg_val = self.__sum_val / self.__count_val
        self.__max_val = max(val, self.__max_val)
        self.__min_val = min(val, self.__min_val)

    @property
    def value(self):
        return self.__val

    @property
    def average(self):
        return self.__avg_val

    @property
    def sum(self):
        return self.__sum_val

    @property
    def max(self):
        return self.__max_val

    @property
    def min(self):
        return self.__min_val

    def __str__(self):
        fmtstr = '{name} {value' + self.fmt + '} ({average' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = ProgressMeter.get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print_meter(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class DictAvergeMeter(object):
    """ Computes and stores the average and current value"""

    def __init__(self):
        self.__key_list = None

    def reset(self):
        if self.__key_list is not None:
            self.__val = {key: 0. for key in self.__key_list}
            self.__avg_val = {key: 0. for key in self.__key_list}
            self.__sum_val = {key: 0. for key in self.__key_list}
            self.__count_val = {key: 0 for key in self.__key_list}
            self.__max_val = {key: -1e12 for key in self.__key_list}
            self.__min_val = {key: 1e12 for key in self.__key_list}

    def update(self, val_dict, n_dict=1):
        """
        Notice that values in val_dict must be average value !
        """
        if self.__key_list is None:
            self.__key_list = val_dict.keys()
            self.reset()

        if isinstance(n_dict, (int, float)):
            new_n_dict = {k: n_dict for k in val_dict.keys()}
            n_dict = new_n_dict

        self.__val = val_dict
        for k in val_dict.keys():
            self.__sum_val[k] += val_dict[k] * n_dict[k]
            self.__count_val[k] += n_dict[k]
            self.__avg_val[k] = self.__sum_val[k] / self.__count_val[k]
            self.__max_val[k] = max(val_dict[k], self.__max_val[k])
            self.__min_val[k] = min(val_dict[k], self.__min_val[k])

    @property
    def sum(self):
        return self.__sum_val

    @property
    def averge(self):
        return self.__avg_val

    @property
    def max(self):
        return self.__max_val

    @property
    def min(self):
        return self.__min_val

    @property
    def values(self):
        return self.__val

    @property
    def count(self):
        return self.__count_val

    def get_value(self, key, mode='value'):
        if mode in ('value', 'val'):
            return self.__val[key]
        elif mode == 'sum':
            return self.__sum_val[key]
        elif mode == 'max':
            return self.__max_val[key]
        elif mode == 'min':
            return self.__min_val[key]
        else:
            return self.__avg_val[key]

    def info(self):
        str_info = '{'
        for k, v in self.__avg_val.items():
            str_info += '{}: {:.4f}, '.format(k, v)

        return str_info.rstrip(', ') + '}'


class ListAverageMeter(object):
    """ Computes and stores the average and current value"""

    def __init__(self):
        self.__len = None

    def reset(self):
        if self.__len is not None:
            self.__val = [0.] * self.__len
            self.__avg_val = [0.] * self.__len
            self.__sum_val = [0.] * self.__len
            self.__count_val = [0] * self.__len
            self.__max_val = [-1e12] * self.__len
            self.__min_val = [1e12] * self.__len

    def update(self, val_list, n_list=1):
        """
        Notice that value in val_list must be average value!
        """
        if self.__len is None:
            self.__len = len(val_list)
            self.reset()

        if isinstance(n_list, (int, float)):
            new_n_list = [n_list] * len(val_list)
            n_list = new_n_list

        self.__val = val_list
        for ind, val in enumerate(val_list):
            self.__sum_val[ind] += val * n_list[ind]
            self.__count_val[ind] += n_list[ind]
            self.__avg_val[ind] = self.__sum_val[ind] / self.__count_val[ind]
            self.__max_val[ind] = max(val, self.__max_val[ind])
            self.__min_val[ind] = min(val, self.__min_val[ind])

    @property
    def sum(self):
        return self.__sum_val

    @property
    def averge(self):
        return self.__avg_val

    @property
    def max(self):
        return self.__max_val

    @property
    def min(self):
        return self.__min_val

    @property
    def values(self):
        return self.__val

    @property
    def count(self):
        return self.__count_val

    def get_value(self, index, mode='value'):
        if mode in ('value', 'val'):
            return self.__val[index]
        elif mode == 'sum':
            return self.__sum_val[index]
        elif mode == 'max':
            return self.__max_val[index]
        elif mode == 'min':
            return self.__min_val[index]
        else:
            return self.__avg_val[index]

    def info(self):
        return "[" + ', '.join(['{:.4f}'.format(x) for x in self.__avg_val]) + "]"
