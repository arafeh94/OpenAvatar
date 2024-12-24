from typing import Callable, TypeVar, Generic

T = TypeVar('T')


class LazyLoader(Generic[T]):
    def __init__(self, loader_function: Callable[[], T], force_load=False, *args, **kwargs):
        self.__loaded: T | None = None
        self.__load_func = loader_function
        self.__args = args
        self.__kwargs = kwargs
        if force_load:
            self.__load()

    def __load(self):
        # noinspection PyArgumentList
        self.__loaded = self.__load_func(*self.__args, **self.__kwargs)

    def get(self) -> T:
        if self.__loaded:
            return self.__loaded
        self.__load()
        return self.get()
