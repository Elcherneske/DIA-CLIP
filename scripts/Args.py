import os
import configparser
import argparse
class Args:
    def __init__(self, config_path = './setting.config'):
        self.config_path = config_path
        if os.path.exists(self.config_path):
            self.config = configparser.ConfigParser()
            self.config.read(self.config_path)
        else:
            print(f"Config file {self.config_path} does not exist, create new config file at {self.config_path}")
    
    def get_config(self, section, option, default=None):
        if section in self.config and option in self.config[section]:
            value = self.config[section][option]
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            try:
                value = value.strip()
                if any(c in value for c in ('.', 'e', 'E')):
                    return float(value)
                return int(value)
            except ValueError:
                return value
        return default
    
    def save_config(self):
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

    def set_config(self, section, option, value, is_save = True):
        if section not in self.config:
            self.config.add_section(section)
        self.config[section][option] = str(value)
        if is_save:
            self.save_config()

def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file')
    return parser