import os

api_keys = "arafeh"


def add_login(fun):
    global api_keys
    if api_keys is None:
        raise Exception("No API key provided")
    if api_keys is not None and api_keys == "":
        raise Exception("API Key is empty")
    if api_keys and api_keys != "arafeh":
        raise Exception("API Key is invalid")
    return fun


@add_login
def get_info():
    return "info"


print(get_info())
