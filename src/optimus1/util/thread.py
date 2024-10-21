import threading
from typing import Any, Callable


class MultiThreadServerAPI(threading.Thread):
    result: Any | None

    def __init__(self, func: Callable, args):
        """
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        """
        threading.Thread.__init__(self)  # 不要忘记调用Thread的初始化方法
        self.func = func
        self.args = args

        # call self.is_alive() to check if the thread is still running
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result
