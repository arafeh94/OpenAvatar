import json
import time
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List

from src.core.interfaces.Pipe import Pipe


class Pipeline:
    def __init__(self, context, pipes=None):
        self.pipes: List[Pipe] = []
        if pipes is not None:
            [self.queue(p) for p in pipes]
        self.timings = []
        self.round = 0
        self.close = False

    def queue(self, pipe: Pipe):
        self.pipes.append(pipe)

    def process(self, obj, middlewares=None):
        if not middlewares:
            middlewares = []
        if not isinstance(middlewares, list):
            middlewares = [middlewares]
        self.timings.append(defaultdict(float))
        flow = [obj]
        for pipe in self.pipes:
            if self.close:
                break
            start = time.time()
            for middleware in middlewares:
                flow.append(middleware({'_': flow[-1], 'flow': flow, 'current_pipe': pipe}))
            flow.append(pipe.exec(flow[-1], flow))
            self.timings[self.round][pipe.__class__.__name__] = time.time() - start
        self.round += 1
        return flow[-1]

    def loop(self):

        def middle_close(args):
            if args == 'stop':
                self.close = True
            return args

        while True:
            if self.close:
                break
            self.process(None, middle_close)

    def execution_times(self):
        return self.timings
