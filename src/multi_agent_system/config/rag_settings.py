import json

class Settings:
    def __init__(self, config_path="../src/multi_agent_system/config/rag_config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.qdrant_url = config.get("qdrant_url")
        self.chunk_size = config.get("chunk_size")
        self.chunk_overlap = config.get("chunk_overlap")
        self.top_n_semantic = config.get("top_n_semantic")
        self.top_n_text = config.get("top_n_text")
        self.alpha = config.get("alpha")
        self.text_boost = config.get("text_boost")
        self.final_top_k = config.get("final_top_k")
        self.mmr = config.get("mmr")
        self.mmr_lambda = config.get("mmr_lambda")