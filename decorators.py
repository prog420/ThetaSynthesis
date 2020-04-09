from time import time


def timer(func):
    def inner(*args, **kwargs):
        t = time()
        result = func(*args, **kwargs)
        print(f'Result of execution of {func.__name__} = {time() - t} seconds')
        return result
    return inner
