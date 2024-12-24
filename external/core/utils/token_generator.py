import random
import string


def generate_token(length: int = 32, chars: str = string.ascii_letters + string.digits) -> str:
    """
    Generate a random token.

    Args:
        length (int): Length of the token. Default is 32.
        chars (str): Characters to use for the token. Default is alphanumeric.

    Returns:
        str: The generated random token.
    """
    return ''.join(random.choices(chars, k=length))
