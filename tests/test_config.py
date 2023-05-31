from utils.config import Config

if __name__ == '__main__':
    configs = Config("test_config.yaml")
    print(configs.param)
    print(configs.param['user1'])