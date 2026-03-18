from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    asr_model: str = "large-v3-turbo"
    asr_model_path: str = "/root/.cache/huggingface"
    asr_device: str = "cuda"
    compute_type: str = "float16"  # INT8 broken on Blackwell sm_120
    model_idle_timeout: int = 0  # seconds; 0 = never unload
    host: str = "0.0.0.0"
    port: int = 9000


settings = Settings()
