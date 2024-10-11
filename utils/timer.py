import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用原函数
        end_time = time.time()  # 记录结束时间
        print(f"{func.__name__} running time: {end_time - start_time:.4f} s")
        return result  # 返回原函数的返回值
    return wrapper