import sys
import logging
from termcolor import colored
import torch.distributed as dist

logger_initialized = {}


LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


DEFAULT_LOG_LEVEL = LOG_LEVEL_DICT['debug']


def get_logger(name,
               log_file=None,
               log_level=logging.INFO,
               is_disp_in_terminal=True,
               is_color=True,
               **kwargs):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # asctime: system_time, name: logger_name, levelname: log_level, message: print_message
    log_formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if is_color:
        disp_formatter = _ColorfulFormatter(log_formatter)
    else:
        disp_formatter = logging.Formatter(log_formatter)

    # mmdetection默认可以在终端打印, 这里为了实际开发, 设置开关
    if is_disp_in_terminal:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(DEFAULT_LOG_LEVEL)
        ch.setFormatter(disp_formatter)
        logger.addHandler(ch)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        fh = logging.FileHandler(log_file, 'w')
        fh.setLevel(DEFAULT_LOG_LEVEL)
        fh.setFormatter(logging.Formatter(log_formatter))
        logger.addHandler(fh)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


class _ColorfulFormatter(logging.Formatter):

    def __init__(self, log_format):
        super(_ColorfulFormatter, self).__init__(log_format)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            color_log = colored("WARNING:" + log, "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR:
            color_log = colored("ERROR:" + log, "red", attrs=["blink"])
        elif record.levelno == logging.CRITICAL:
            color_log = colored("CRITICAL:" + log, "red", attrs=["blink"])
        elif record.levelno == logging.INFO:
            color_log = colored("INFO:" + log, "green", attrs=["blink"])
        else:
            return log

        return color_log