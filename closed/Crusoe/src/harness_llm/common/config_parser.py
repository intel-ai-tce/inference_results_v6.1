import sys
from harness_llm.common.container_utils import print_dict
from omegaconf import OmegaConf
from pathlib import Path


class HarnessCfg:

    def __init__(self):
        self.base_config = OmegaConf.load(Path(__file__).parent / "config.yaml")
        self.model_cfg = None
        self.config = None
        self.model_declares_sglang_engine = False

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

        raw_model_cfg = OmegaConf.load(config_path + "/" + config_name + ".yaml")
        # Remember which engine the model yaml actually declares. The base config
        # carries defaults for BOTH engines, so this must be read from the raw
        # model yaml (before merging) to tell vllm and sglang apart.
        self.model_declares_sglang_engine = "sglang_engine_config" in raw_model_cfg
        self.model_cfg = OmegaConf.merge(self.base_config, raw_model_cfg)

        OmegaConf.set_struct(self.model_cfg.harness_config, True)

        self.config = OmegaConf.merge(self.model_cfg, config_overrides)

        if self.config.backend == 'MISSING':
            print("backend is not set, fallback to vllm", file=sys.stderr)
            self.config.backend = 'vllm'

        # The zmq transport can drive either the vllm or the sglang engine. The
        # engine is picked by whichever *_engine_config the model yaml provides
        # (deepseek-r1 ships sglang_engine_config, gpt-oss ships
        # vllm_engine_config). vllm/ray always use the vllm engine.
        uses_sglang_engine = ("sglang" == self.config.backend) or (
            "zmq" == self.config.backend
            and self.model_declares_sglang_engine
        )

        if self.config.backend in ("vllm", "ray", "zmq") and not uses_sglang_engine:
            self.select_engine_config(engine="vllm")

        elif uses_sglang_engine:
            self.select_engine_config(engine="sglang")

        else:
            print("backend is not set", file=sys.stderr)
            sys.exit(1)

        return self

    def select_engine_config(self, engine: str):
        """Promote the chosen engine's *_engine_config / *_sampling_config /
        *_env_config to the generic keys used by the harness and drop the other
        engine's config so nothing stale leaks through."""
        engines = {"vllm", "sglang"}
        assert engine in engines, f"Unknown engine {engine}"
        other = (engines - {engine}).pop()

        self.merge_env_configs(f"{engine}_env_config")
        self.rename_key(f"{engine}_engine_config", "llm_config")
        self.rename_key(f"{engine}_sampling_config", "sampling_params")

        for suffix in ("engine_config", "sampling_config", "env_config"):
            key = f"{other}_{suffix}"
            if key in self.config:
                del self.config[key]


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
