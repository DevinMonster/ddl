import os.path

import yaml


class Config:
    """
    读取file_path("./xxx/config.yaml")配置文件
    """
    def __init__(self, file_path: str):
        assert file_path.endswith(".yaml") and os.path.exists(file_path), \
            f"configration file doesn't exists at {file_path}!"
        # read yaml
        print("Reading Parameters...")
        with open(file_path, 'r', encoding='utf-8') as f:
            _config_str = f.read()
        self._param = yaml.load(_config_str, yaml.FullLoader)
        print("Parameters done!")

    @property
    def param(self):
        return self._param