
import argparse
import yaml
config = {}


def _init():
    """
    Initialize config from YAML file.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--f', type=str, default="D:\Project\SSVEP_AUG\etc\config.yaml", help='--config file')

    opt = parser.parse_args()
    config_path = opt.f
    global config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

def get_global_conf():
    _init()
    # Return global config; on failure print read error
    try:
        return config
    except:
        print('Failed to read config.\n')
        return {}

config = get_global_conf()