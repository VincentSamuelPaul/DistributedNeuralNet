import yaml
import os

def load_config(path: str = None) -> dict:
    if path is None:
        path = os.environ.get("CONFIG_PATH", "/app/config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)