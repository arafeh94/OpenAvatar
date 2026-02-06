import json
import logging
from abc import abstractmethod
from typing import List, Optional, TYPE_CHECKING

from easydict import EasyDict

from core.tools import utils
from manifest import Manifest

if TYPE_CHECKING:
    from services.rtc.src.peer import ServerPeer


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


class ToolRequest:
    def __init__(self, **kwargs):
        self.id = None
        self.__dict__.update(kwargs)

    @abstractmethod
    def process(self, peer: 'ServerPeer'):
        pass

    def packet(self, payload, status='200'):
        return Packet(self.id, payload, status)

    def end_packet(self):
        return Packet(self.id, None, '204')



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
        except json.decoder.JSONDecodeError as json_error:
            self.logger.error("Exception while parsing json request: {}".format(json_error))
        except Exception as dict_parse_error:
            self.logger.error("Exception while converting json to dict: {}".format(dict_parse_error))

    def tools(self):
        return list(self.payload().keys())

    def parse_tools(self) -> List[ToolRequest]:
        tool_requests = []
        rtc_tools = Manifest().query('rtc.tools')
        for tool in self.tools():
            if tool in rtc_tools:
                _request = utils.get_class_instance(rtc_tools[tool], **{**self.payload().get(tool), 'id': self.__id})
                tool_requests.append(_request)
        return tool_requests

    def payload(self) -> Optional[EasyDict]:
        if '_parsed_payload' in self.__dict__:
            return self._parsed_payload
        if isinstance(self.__payload, str):
            if self.__type == 'json':
                self._parsed_payload = EasyDict(json.loads(self.__payload))
        elif isinstance(self.__payload, dict):
            self._parsed_payload = EasyDict(self.__payload)
        else:
            self._parsed_payload = self.__payload
        return self._parsed_payload


if __name__ == '__main__':
    request = Requests('{"fake":{"data":"hello"}')
    print(request.tools())
    print(request.parse_tools())
    for request in request.parse_tools():
        # noinspection PyTypeChecker
        request.process(None)
