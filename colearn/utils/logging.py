import logging
import sys
import click


""" logging helpers, inspired from https://github.com/fetchai/agents-aea/blob/master/aea/cli/utils/loggers.py """


class ColorFormatter(logging.Formatter):
    """The default formatter for cli output."""

    colors = {
        "error": dict(fg="red"),
        "exception": dict(fg="red"),
        "critical": dict(fg="red"),
        "debug": dict(fg="blue"),
        "info": dict(fg="green"),
        "warning": dict(fg="yellow"),
    }

    def format(self, record):
        """Format the log message."""
        if not record.exc_info:
            level = record.levelname.lower()
            msg = record.getMessage()
            if level in self.colors:
                prefix = click.style("{}: ".format(level), **self.colors[level])
                msg = "\n".join(prefix + x for x in msg.splitlines())
            return msg
        return logging.Formatter.format(self, record)  # pragma: no cover


def default_logging_config(logger):  # pylint: disable=redefined-outer-name
    """Set up the default handler and formatter on the given logger."""
    default_handler = logging.StreamHandler(stream=sys.stdout)
    default_handler.formatter = ColorFormatter()
    logger.handlers = [default_handler]
    logger.propagate = True
    return logger


_log_levels = {}  # pylint: disable=W0603  # type: ignore
_loggers = {}  # pylint: disable=W0603  # type: ignore


def _set_logger_level(logger, log_level):
    level = logging.getLevelName(log_level.upper())
    logger.setLevel(level)


def _update_log_level(logger_name, logger):
    if logger_name in _log_levels:
        _set_logger_level(logger, _log_levels[logger_name])
    elif "default" in _log_levels:
        _set_logger_level(logger, _log_levels["default"])


def get_logger(name, name_length=1):
    global _loggers  # pylint: disable=W0603
    splitted = name.split(".")
    logger_name = ".".join(splitted[-name_length:])
    logger = logging.getLogger(logger_name)
    logger = default_logging_config(logger)
    _update_log_level(logger_name, logger)
    _loggers[logger_name] = logger
    return logger


def set_log_levels(config):
    global _log_levels, _loggers  # pylint: disable=W0603
    _log_levels = {**config}
    for name, logger in _loggers.items():
        _update_log_level(name, logger)
