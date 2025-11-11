import importlib, yaml
from pathlib import Path
from src.io.store import make_store


class Registry:
    def __init__(self, cfg_path="configs/datasources.yaml"):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())

    @property
    def bucket_uri(self):
        return self.cfg["storage"]["bucket"].rstrip("/")

    @property
    def bronze_prefix(self):
        return self.cfg["storage"]["bronze_prefix"].strip("/")

    @property
    def silver_prefix(self):
        return self.cfg["storage"]["silver_prefix"].strip("/")
    
    @property
    def gold_prefix(self):
        return self.cfg["storage"]["gold_prefix"].strip("/")

    @property
    def assets(self):
        return self.cfg["assets"]

    @property
    def backfill_window(self):
        b = self.cfg.get("backfill", {})
        return b.get("start"), b.get("end")

    def adapter_for(self, provider_name: str):
        prov = self.cfg["providers"][provider_name]
        module = importlib.import_module(f"src.{prov['module']}")
        return module
    
    def make_store(self):
        return make_store(self.cfg)
