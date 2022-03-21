# import shutil
from typing import Dict, Any
import os
import yaml


def ParseYAML(yaml_file: str) -> Dict[str, Any]:
    file_handle = open(yaml_file, "r", encoding="utf-8")
    params = yaml.load(file_handle, Loader=yaml.FullLoader)
    if params["input_bands"] == 3:  # TODO: 这里为啥
        params["mean"] = (0.472455, 0.320782, 0.318403)
        params["std"] = (0.215084, 0.408135, 0.409993)
    else:
        params["mean"] = (0.472455, 0.320782, 0.318403, 0.357)
        params["std"] = (0.215084, 0.408135, 0.409993, 0.195)

    params["save_dir"] = \
        os.path.join(params["root_path"], f"exp({params['exp_name']})")
    params["save_dir_model"] = \
        os.path.join(params["save_dir"],
                     params["model_name"]+"_"+params["model_version"])
    #shutil.copy(yaml_file, params['save_dir_model'])
    return params


if __name__ == "__main__":
    """
    应对直接被执行。
    """
    params = ParseYAML("config.yaml")
    print(params)
