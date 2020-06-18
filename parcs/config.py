import configparser

from .network_utils import *

class Config:
    NODE_SECTION = 'Node'
    MASTER_NODE_SECTION = 'Master Node'

    def __init__(self, ip, port, master_ip=None, master_port=None):
        self.master = master_ip is None
        self.ip = ip if ip else get_ip()
        self.port = port if port else find_free_port()
        self.master_ip = master_ip
        self.master_port = master_port

    @staticmethod
    def load_from_file(config_path):
        configuration = configparser.ConfigParser()
        configuration.read(config_path)

        master = configuration.getboolean(Config.NODE_SECTION, 'master')
        ip = configuration.get(Config.NODE_SECTION, 'ip') if configuration.has_option(Config.NODE_SECTION,
                                                                                             'ip') else None
        port = configuration.getint(Config.NODE_SECTION, 'port') if configuration.has_option(Config.NODE_SECTION,
                                                                                             'port') else None

        if not master:
            master_ip = configuration.get(Config.MASTER_NODE_SECTION, 'ip')
            master_port = configuration.getint(Config.MASTER_NODE_SECTION, 'port')
            return Config(ip, port, master_ip, master_port)
        return Config(ip, port)
