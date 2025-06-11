import logging
from typing import Literal, Optional, Union

__all__ = ["get_logger", "set_logger_level", "set_logger_file"]


class LoggerSingleton:
    """A singleton wrapper for the 'worldVacBackend' logger."""

    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = cls._create_logger()
        return cls._logger

    @staticmethod
    def _create_logger():
        logger = logging.getLogger("worldVacBackend")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Avoid duplicate output by disabling propagation

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger


def get_logger():
    """Retrieve the singleton logger instance."""
    return LoggerSingleton.get_logger()


def set_logger_level(
    level: Union[
        Literal["DEBUG"],
        Literal["INFO"],
        Literal["WARNING"],
        Literal["WARN"],
        Literal["ERROR"],
        Literal["CRITICAL"],
        Literal["FATAL"],
    ],
):
    """Set the logger's level."""
    get_logger().setLevel(level)


def set_logger_file(
    filename: str,
    mode: str = "a",
    encoding: Optional[str] = None,
    delay: bool = False,
    errors: Optional[str] = None,
):
    """Set the logger's file."""
    handler = logging.FileHandler(filename, mode, encoding, delay, errors)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    get_logger().addHandler(handler)
