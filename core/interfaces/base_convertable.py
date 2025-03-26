import io
import typing


class Convertable:

    def as_byte_buffer(self) -> io.BytesIO:
        pass

    def as_streaming_response(self) -> typing.Any:
        pass

    def as_file(self, filename) -> bool:
        pass
