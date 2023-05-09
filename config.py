# Importing all needed modules.
from configparser import ConfigParser


def get_configurations(filename : str = "config.ini") -> dict:
    '''
        This function reads the configuration file and loads them into a dictionary.
            :param filename: str, default = 'config.ini'
                The path to the file with configurations.
            :return: dict
                The file with configurations.
    '''
    # Parsing of the configuration file.
    parser = ConfigParser()
    parser.read(filename)

    # Filling the configuration dictionary with sections.
    config_dict = dict()
    for section in parser.sections():
        # Loading the configurations for the section.
        config = dict()
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]

        # Adding the section to the configuration dictionary.
        config_dict[section] = config
    return config_dict

class BaseConfig:
    pass


class Service:
    def __init__(self, service_config : dict) -> None:
        '''
            The constructor of the Service Config.
                :param service_config: dict
                    The configuration of a specific service in the cluster.
                    Usually containing the host, port, access token and endpoints.
        '''
        self.host = service_config["host"]
        self.port = service_config["port"]
        self.secret_key = service_config["secret-key"]

        # Setting the endpoints as object fields.
        for key in service_config:
            if key.endswith("endpoint"):
                setattr(self, key.replace("-", "_"), service_config[key])


class ConfigManager:
    def __init__(self, filename : str) -> None:
        '''
            The constructor of the Configuration Manager class.
                :param filename: str
                    The path to the file with configurations.
        '''
        # Getting the configuration dictionary for the configuration file.
        self.general_configs = []
        config_dict = get_configurations(filename)

        # Parsing the configs extracted.
        for config in config_dict:
            if config.startswith("service"):
                self.add_service_config(config, config_dict[config])
            elif config.endswith("dict"):
                self.generate_config(config, config_dict[config], dictionary=True)
            else:
                self.generate_config(config, config_dict[config])

    def add_service_config(self, service_name : str, config_dict : dict) -> None:
        '''
            This function handles the configurations for services.
                :param service_name: str
                    The name of the service which configurations are these.
                :param config_dict: dict
                    The configuration of the service.
        '''
        # Setting the service configurations as a Service instance.
        setattr(
            self,
            service_name.replace("-", "_"),
            Service(config_dict),
        )

    def generate_config(self, config_name : str, config_dict : dict, dictionary : bool = False) -> None:
        '''
            This function converts an dictionary config into a BaseConfig object.
                :param config_name: str
                    The name of configuration.
                :param config_dict: : dict
                    The configurations dictionary.
                :param dictionary: bool, default = False
                    If False a BaseConfig will be created.
                    Else the configuration will be saved as a dictionary.
        '''
        # Appending the configuration mame to the general ones.
        self.general_configs.append(config_name)

        if not dictionary:
            # Setting the configurations as a BaseConfig object.
            setattr(self, config_name.replace("-", "_"), BaseConfig())
            for key in config_dict:
                if config_dict[key].replace(".", "").isnumeric() and config_dict[key].count(".")<=1:
                    value = float(config_dict[key]) if "." in config_dict[key] else int(config_dict[key])
                else:
                    value = config_dict[key]
                setattr(
                    getattr(self, config_name.replace("-", "_")), key,
                    value
                )
        else:
            # Setting the configurations as a dictionary.
            setattr(self, config_name.replace("-", "_"), config_dict)
            for key in getattr(self, config_name.replace("-", "_")):
                if getattr(self, config_name.replace("-", "_"))[key].replace(".", "").isnumeric() and getattr(self, config_name.replace("-", "_"))[key].count(".")<=1:
                    value = float(getattr(self, config_name.replace("-", "_"))[key]) if "." in getattr(self, config_name.replace("-", "_"))[key] \
                        else int(getattr(self, config_name.replace("-", "_"))[key])
                    getattr(self, config_name.replace("-", "_"))[key] = value

    def generate_info_for_service_discovery(self, interest_config=None):
        '''
            THis function returns the general configurations as a dictionary.
        '''
        # Creating a empty dictionary and adding the configurations into it.
        if interest_config is None:
            interest_config = ["general", "security"]
        service_information = dict()
        for personal_config in self.general_configs:
            if personal_config in interest_config:
                if type(getattr(self, personal_config.replace("-", "_"))) is dict:
                    service_information[personal_config] = getattr(self, personal_config.replace("-", "_"))
                else:
                    service_information[personal_config] = getattr(self, personal_config.replace("-", "_")).__dict__
        return service_information