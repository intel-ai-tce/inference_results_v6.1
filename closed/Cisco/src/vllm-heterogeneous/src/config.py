import sys
from pathlib import Path
from omegaconf import OmegaConf

from config_helpers import select_model_profile

BASE_CONFIG = Path(__file__).parent.parent / "config" / "base.yaml"


def _print_dict(d, indent=0):
    for key, value in d.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_dict(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}: {', '.join(map(str, value))}")
        else:
            print(f"{prefix}{key}: {value}")


class HarnessCfg:

    def __init__(self):
        self.base_config = OmegaConf.load(BASE_CONFIG)
        self.config = None

    def __getattr__(self, name):
        if name in ("base_config", "config"):
            raise AttributeError(name)
        if self.config is not None and name in self.config:
            return self.config[name]
        raise AttributeError(f"'{name}' not found in config")

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.config[key] = value

    def create_from_cli(self):
        args = sys.argv[1:]
        parsed = []
        key = None
        value = None

        for arg in args:
            if arg.startswith("--"):
                stripped = arg[2:]
                if "=" in stripped:
                    key, value = stripped.split("=", 1)
                else:
                    key = stripped
                    value = None
            elif "=" in arg:
                parsed.append(arg)
            elif "=" not in arg:
                value = arg

            if None not in (key, value):
                key = key.replace("-", "_")
                parsed.append(f"{key}={value}")
                key = None
                value = None

        return self.create(OmegaConf.from_dotlist(parsed))

    def create(self, overrides):
        config_path = OmegaConf.select(overrides, "config_path")
        config_name = OmegaConf.select(overrides, "config_name")

        if None in (config_path, config_name):
            print("config_path/config_name are missing", file=sys.stderr)
            sys.exit(1)

        model_cfg = OmegaConf.load(config_path + "/" + config_name + ".yaml")
        profile = OmegaConf.select(overrides, "profile", default=None)
        if profile:
            try:
                model_cfg = OmegaConf.create(select_model_profile(
                    OmegaConf.to_container(model_cfg, resolve=False),
                    str(profile),
                    OmegaConf.select(overrides, "scenario", default=None),
                ))
            except ValueError as exc:
                print(f"invalid model profile: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            OmegaConf.set_struct(model_cfg, False)
            if "profiles" in model_cfg:
                del model_cfg["profiles"]
        OmegaConf.set_struct(overrides, False)
        if "profile" in overrides:
            del overrides["profile"]
        model_cfg = OmegaConf.merge(self.base_config, model_cfg)
        OmegaConf.set_struct(model_cfg.harness_config, True)
        self.config = OmegaConf.merge(model_cfg, overrides)

        if self.config.backend == "MISSING":
            self.config.backend = "pd"

        self._apply_hardware_overrides()
        self._merge_env("vllm_env_config")
        self._rename("vllm_engine_config", "llm_config")
        self._rename("vllm_sampling_config", "sampling_params")

        return self

    def _apply_hardware_overrides(self):
        """Merge per-hardware sub-keys into their parent sections.

        If ``hardware`` is set (via YAML or CLI, e.g. ``hardware=h200``),
        each config section that contains a matching sub-key gets those
        values merged on top.  All hardware sub-keys (dicts whose key
        matches a known hardware name) are then removed so downstream
        consumers see a flat section.
        """
        hw = OmegaConf.select(self.config, "hardware", default=None)
        if not hw:
            return
        
        
        
        
        scenario = OmegaConf.select(self.config, "scenario", default=None)
        scenario = str(scenario).lower() if scenario else None
        sections = (
            "pd_config", "vllm_engine_config", "vllm_env_config",
            "prefiller", "decoder", "standalone", "standalone_mp",
        )
        for section_name in sections:
            section = OmegaConf.select(self.config, section_name, default=None)
            if section is None:
                continue
            sub = OmegaConf.select(section, hw, default=None)
            if sub is not None and OmegaConf.is_dict(sub):
                OmegaConf.set_struct(section, False)
                self.config[section_name] = OmegaConf.merge(section, sub)
                section = self.config[section_name]
            if scenario:
                scn = OmegaConf.select(section, scenario, default=None)
                if scn is not None and OmegaConf.is_dict(scn):
                    OmegaConf.set_struct(section, False)
                    self.config[section_name] = OmegaConf.merge(section, scn)
                    section = self.config[section_name]
            OmegaConf.set_struct(section, False)
            for k in list(section):
                if OmegaConf.is_dict(OmegaConf.select(section, k, default=None)):
                    del section[k]

    def _merge_env(self, key):
        if key in self.config and self.config[key]:
            self.config.env_config = OmegaConf.merge(
                self.config.env_config, self.config[key])
            del self.config[key]

    def _rename(self, old, new):
        if old in self.config:
            self.config[new] = self.config[old]
            del self.config[old]

    def print_config(self):
        _print_dict(OmegaConf.to_object(self.config), indent=0)

    def get_with_default(self, key, default):
        return self.config.get(key, default)
