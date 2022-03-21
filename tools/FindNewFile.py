import os
from typing import Union


def FindNewFile(dir: str) -> Union[None,str]:
    file_list = os.listdir(dir).sort(key=lambda fn: os.path.getmtime(
        dir + fn) if not os.path.isdir(dir + fn) else 0)
    if file_list is None or len(file_list) == 0:
        return None
    return os.path.join(dir, file_list[-1])
