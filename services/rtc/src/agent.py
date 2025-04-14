import json
import logging
from abc import abstractmethod
from typing import List

from easydict import EasyDict

from core.tools import utils
from manifest import Manifest
from services.rtc.src.typing import ServerPeer


class AgentRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def is_valid(self):
        for k, v in self.__dict__.items():
            if v is None and not k.startswith('_'):
                return False
        return True

    @abstractmethod
    def process(self, peer: 'ServerPeer'):
        pass


class Requests:
    def __init__(self, message):
        self.is_valid = False
        self.data = None
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.data = json.loads(message)
            self.data = EasyDict(self.data)
            self.is_valid = True
        except json.decoder.JSONDecodeError as json_error:
            self.logger.error("Exception while parsing json request: {}".format(json_error))
        except Exception as dict_parse_error:
            self.logger.error("Exception while converting json to dict: {}".format(dict_parse_error))

    def agents(self):
        if not self.is_valid:
            return None
        return list(self.data.keys())

    def parse_agents(self) -> List[AgentRequest]:
        agent_requests = []
        rtc_agents = Manifest().query('rtc.agents')
        for agent in self.agents():
            if agent in rtc_agents:
                request = utils.get_class_instance(rtc_agents[agent], **self.data.get(agent))
                agent_requests.append(request)
        return agent_requests


if __name__ == '__main__':
    request = Requests('{"fake":{"data":"hello"}')
    # request = Requests("{'avatar':{'repeat':'hello samira how are you today', 'persona':'lisa_casual_720_pl'}}")
    print(request.agents())
    print(request.parse_agents())
    for request in request.parse_agents():
        request.process(None)
