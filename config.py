"""
Configuration loader for the Offline Subtitle Generator.
Loads from config.yaml and allows CLI argument overrides.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_s16le"


@dataclass
class VADConfig:
    threshold: float = 0.35
    min_speech_sec: float = 0.15
    min_silence_sec: float = 0.3
    padding_sec: float = 0.3
    energy_threshold: float = 0.001


@dataclass
class ASRConfig:
    model: str = "small"
    compute_type: str = "int8"
    beam_size: int = 3
    threads: int = 0  # 0 = auto-detect CPU cores
    language: Optional[str] = None
    max_segment_sec: int = 30
    best_of: int = 1
    patience: float = 1.0
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    min_confidence: float = -1.0
    word_timestamps: bool = True
    temperature: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    initial_prompt: Optional[str] = None


@dataclass
class SEDConfig:
    model_path: str = "models/yamnet.tflite"
    class_map_path: str = "models/yamnet_class_map.csv"
    min_confidence: float = 0.4
    frame_duration: float = 0.96
    hop_duration: float = 0.48
    run_on_speech: bool = False
    max_gap_merge: float = 0.5


@dataclass
class MergeConfig:
    min_duration: float = 0.8
    max_duration: float = 7.0
    merge_gap: float = 0.5


@dataclass
class ThreadingConfig:
    mode: str = "parallel"
    max_cpu_percent: int = 70
    throttle_check_interval: float = 2.0


@dataclass
class CacheConfig:
    enabled: bool = True
    directory: str = "cache"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class AppConfig:
    """Top-level application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    sed: SEDConfig = field(default_factory=SEDConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    threading: ThreadingConfig = field(default_factory=ThreadingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def threading_mode(self) -> str:
        return self.threading.mode

    @property
    def max_cpu_percent(self) -> int:
        return self.threading.max_cpu_percent

    def update_from_args(self, args):
        """Override config values from CLI arguments."""
        if hasattr(args, "model") and args.model:
            self.asr.model = args.model
        if hasattr(args, "parallel") and args.parallel:
            self.threading.mode = "parallel"
        if hasattr(args, "max_cpu") and args.max_cpu:
            self.threading.max_cpu_percent = args.max_cpu
        if hasattr(args, "language") and args.language:
            self.asr.language = args.language


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from a YAML file.
    Falls back to defaults if file is missing.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(f"Config file not found at {path}, using defaults.")
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = AppConfig(
        audio=_dict_to_dataclass(AudioConfig, raw.get("audio")),
        vad=_dict_to_dataclass(VADConfig, raw.get("vad")),
        asr=_dict_to_dataclass(ASRConfig, raw.get("asr")),
        sed=_dict_to_dataclass(SEDConfig, raw.get("sed")),
        merge=_dict_to_dataclass(MergeConfig, raw.get("merge")),
        threading=_dict_to_dataclass(ThreadingConfig, raw.get("threading")),
        cache=_dict_to_dataclass(CacheConfig, raw.get("cache")),
        logging=_dict_to_dataclass(LoggingConfig, raw.get("logging")),
    )

    logger.info(f"Configuration loaded from {path}")
    return config
