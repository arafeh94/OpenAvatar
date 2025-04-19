import json
import logging
from abc import abstractmethod
from typing import List, Optional

from easydict import EasyDict

from core.tools import utils
from manifest import Manifest
from services.rtc.src.typing import ServerPeer


class Packet:
    def __init__(self, id, payload, status=None):
        self.__id = id
        self.__payload = payload
        self.__status = status

    @property
    def id(self):
        return self.__id

    @property
    def payload(self):
        return self.__payload

    def as_json(self):
        return json.dumps({'id': self.id, 'payload': self.payload, 'status': self.__status})


class AgentRequest:
    def __init__(self, **kwargs):
        self.id = None
        self.__dict__.update(kwargs)

    def is_valid(self):
        for k, v in self.__dict__.items():
            if v is None and not k.startswith('_'):
                return False
        return True

    @abstractmethod
    def process(self, peer: 'ServerPeer'):
        pass

    def packet(self, payload):
        return Packet(self.id, payload)

    def end_packet(self):
        return Packet(self.id, None, 'ended')


class Requests:
    def __init__(self, message):
        self.is_valid = False
        self.__payload = None
        self.__id = None
        self.__type = None
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.data = json.loads(message)
            self.data = EasyDict(self.data)
            self.__id = self.data['id']
            self.__type = self.data['type']
            self.__payload = self.data['payload']
            self._parsed_payload = self.payload()
            self.is_valid = True
        except json.decoder.JSONDecodeError as json_error:
            self.logger.error("Exception while parsing json request: {}".format(json_error))
        except Exception as dict_parse_error:
            self.logger.error("Exception while converting json to dict: {}".format(dict_parse_error))

    def agents(self):
        if not self.is_valid:
            return None
        return list(self.payload().keys())

    def parse_agents(self) -> List[AgentRequest]:
        agent_requests = []
        rtc_agents = Manifest().query('rtc.agents')
        for agent in self.agents():
            if agent in rtc_agents:
                request = utils.get_class_instance(rtc_agents[agent], **{**self.payload().get(agent), 'id': self.__id})
                agent_requests.append(request)
        return agent_requests

    def payload(self) -> Optional[EasyDict]:
        if '_parsed_payload' in self.__dict__:
            return self._parsed_payload
        if self.__type == 'json':
            self._parsed_payload = EasyDict(json.loads(self.__payload))
            return self._parsed_payload
        return None


if __name__ == '__main__':
    request = Requests('{"fake":{"data":"hello"}')
    print(request.agents())
    print(request.parse_agents())
    for request in request.parse_agents():
        # noinspection PyTypeChecker
        request.process(None)
