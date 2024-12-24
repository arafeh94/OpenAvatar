from abc import ABC, abstractmethod


class BaseService(ABC):
    @abstractmethod
    def exec(self, *args, **kwargs):
        pass

    def extract_first(self, arg_name=None, *args, **kwargs):
        result = kwargs[0] if len(kwargs) > 0 else None
        if not result and arg_name and arg_name not in result:
            raise Exception(f'{arg_name} is not in service arguments, '
                            f'make sure you include "{arg_name}" like llm(prompt=...)')
        result = kwargs[arg_name]
        return result
