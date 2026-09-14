import sys
from harness_llm.common.container_utils import print_dict
from omegaconf import OmegaConf
from pathlib import Path


class HarnessCfg:

    def __init__(self):
        self.base_config = OmegaConf.load(Path(__file__).parent / "config.yaml")
        self.model_cfg = None
        self.config = None

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        else:
            raise AttributeError(f"'{name}' is not a valid attribute")

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        raise KeyError(f"'{key}' is not a valid key")

    def __setitem__(self, key, value):
        self.config[key] = value

    def create_from_cli(self):
        args = sys.argv[1:]

        parsed_args = []
        key = None
        value = None

        for arg in args:
            if arg.startswith("--"):
                stripped_key = arg[2:]
                if '=' in stripped_key:
                    key, value = stripped_key.split('=', 1)
                else:
                    key = stripped_key
                    value = None

            elif '=' in arg:
                parsed_args.append(arg)

            elif '=' not in arg:
                value = arg

            if None not in (key, value):
                key = key.replace('-', '_')
                parsed_args.append(f"{key}={value}")
                key = None
                value = None

        return self.create(OmegaConf.from_dotlist(parsed_args))


    def create_from_optuna(self, config_path, config_name, backend, overrides):
        optuna_conf = OmegaConf.from_dotlist(overrides)
        optuna_conf.config_path = config_path
        optuna_conf.config_name = config_name
        optuna_conf.backend = backend

        return self.create(optuna_conf)


    def create(self, config_overrides):
        config_path = OmegaConf.select(config_overrides, "config_path")
        config_name = OmegaConf.select(config_overrides, "config_name")

        if None in (config_path, config_name):
            print("config_path/config_name are missing", file=sys.stderr)
            sys.exit(1)

        self.model_cfg = OmegaConf.load(config_path + "/" + config_name + ".yaml")
        self.model_cfg = OmegaConf.merge(self.base_config, self.model_cfg)

        OmegaConf.set_struct(self.model_cfg.harness_config, True)

        self.config = OmegaConf.merge(self.model_cfg, config_overrides)

        if self.config.backend == 'MISSING':
            print("backend is not set, fallback to vllm", file=sys.stderr)
            self.config.backend = 'vllm'

        if "vllm" == self.config.backend or "ray" == self.config.backend or "zmq" == self.config.backend:
            self.merge_env_configs("vllm_env_config")
            self.rename_key("vllm_engine_config", "llm_config")
            self.rename_key("vllm_sampling_config", "sampling_params")
            if "sglang_engine_config" in self.config:
                del self.config.sglang_engine_config
            if "sglang_sampling_config" in self.config:
                del self.config.sglang_sampling_config
            if "sglang_env_config" in self.config:
                del self.config.sglang_env_config

        elif "sglang" == self.config.backend:
            self.merge_env_configs("sglang_env_config")
            self.rename_key("sglang_engine_config", "llm_config")
            self.rename_key("sglang_sampling_config", "sampling_params")
            if "vllm_engine_config" in self.config:
                del self.config.vllm_engine_config
            if "vllm_sampling_config" in self.config:
                del self.config.vllm_sampling_config
            if "vllm_env_config" in self.config:
                del self.config.vllm_env_config

        else:
            print("backend is not set", file=sys.stderr)
            sys.exit(1)

        return self


    def print_config(self):
        print_dict(OmegaConf.to_object(self.config), indent=0)

    def merge_env_configs(self, env_conf):
        if self.config[env_conf]:
            self.config.env_config = OmegaConf.merge(self.config.env_config, self.config[env_conf])
            del self.config[env_conf]

    def rename_key(self, origin, target):
        self.config[target] = self.config[origin]
        del self.config[origin]

    def get_with_default(self, key, default):
        return self.config.get(key, default)
