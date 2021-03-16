import time
from functools import wraps

"""
time consume: timefn()函数可以非常方便的核查某个函数的时间消耗, 用法只需要import到指定文件,
然后对需要测试的函数装饰即可.
"""


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: " + fn.__name__ + " took " + str(t2 - t1) + " seconds !")
        return result
    return measure_time


@timefn
def mysum():
    s = 0
    for i in range(1000000):
        s += 1
    return s


if __name__ == "__main__":
    mysum()