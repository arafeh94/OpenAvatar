import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj, display_x=5):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    memory_profile = {}
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                sizeof = sys.getsizeof(obj)
                memory_profile[obj.__class__.__name__] = sizeof
                size += sizeof
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size, top_items(memory_profile, display_x)


def top_items(data_dict, x=5):
    sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
    top_x = dict(sorted_items[:x])
    return top_x
