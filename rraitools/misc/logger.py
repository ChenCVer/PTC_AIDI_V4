import logging
import os
import sys
from termcolor import colored

from ..filetools import FileHelper

LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

DEFAULT_LOG_LEVEL = LOG_LEVEL_DICT['debug']
DEFAULT_LOG_FORMAT = '%(asctime)s%(message)s'


class Logger(object):
    """
    Args:
      Log level: CRITICAL>ERROR>WARNING>INFO>DEBUG.
      log format: The format of log messages.
    """
    logger = None

    @staticmethod
    def init(save_dir=None,
             filename="log.txt",
             is_disp_on_terminal=True,
             is_color=True,
             log_format=DEFAULT_LOG_FORMAT,
             log_level='debug'):

        if Logger.logger is None:
            Logger.__setup_logger(save_dir=save_dir,
                                  filename=filename,
                                  is_disp_on_terminal=is_disp_on_terminal,
                                  is_color=is_color,
                                  log_format=log_format,
                                  log_level=log_level)

        return Logger

    @staticmethod
    def set_level(log_level):
        Logger.logger.setLevel(LOG_LEVEL_DICT[str(log_level).lower()])

    @staticmethod
    def check_logger():
        if Logger.logger is None:
            raise Exception('Logger is None, you should init first!')

    @staticmethod
    def debug(message, use_prefix_fun=True):
        Logger.check_logger()
        if use_prefix_fun:
            # 不能抽取成公共函数，因为系统栈获取的不是真实调用位置的文件名称
            filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
            lineno = sys._getframe().f_back.f_lineno
            prefix = '[{}, {}]'.format(filename, lineno)
            Logger.logger.debug('{} {}'.format(prefix, message))
        else:
            Logger.logger.debug('{}'.format(message))

    @staticmethod
    def info(message, use_prefix_fun=True):
        Logger.check_logger()
        if use_prefix_fun:
            filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
            lineno = sys._getframe().f_back.f_lineno
            prefix = '[{}, {}]'.format(filename, lineno)
            Logger.logger.info('{} {}'.format(prefix, message))
        else:
            Logger.logger.info('{}'.format(message))

    @staticmethod
    def warn(message, use_prefix_fun=True):
        Logger.check_logger()
        if use_prefix_fun:
            filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
            lineno = sys._getframe().f_back.f_lineno
            prefix = '[{}, {}]'.format(filename, lineno)
            Logger.logger.warn('{} {}'.format(prefix, message))
        else:
            Logger.logger.warn('{}'.format(message))

    @staticmethod
    def error(message, use_prefix_fun=True):
        Logger.check_logger()
        if use_prefix_fun:
            filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
            lineno = sys._getframe().f_back.f_lineno
            prefix = '[{}, {}]'.format(filename, lineno)
            Logger.logger.error('{} {}'.format(prefix, message))
        else:
            Logger.logger.error('{}'.format(message))

    @staticmethod
    def critical(message, use_prefix_fun=True):
        Logger.check_logger()
        if use_prefix_fun:
            filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
            lineno = sys._getframe().f_back.f_lineno
            prefix = '[{}, {}]'.format(filename, lineno)
            Logger.logger.critical('{} {}'.format(prefix, message))
        else:
            Logger.logger.critical('{}'.format(message))

    @staticmethod
    def __setup_logger(save_dir,
                       filename,
                       is_disp_on_terminal,
                       is_color,
                       log_format,
                       log_level):

        Logger.logger = logging.getLogger('logger')
        assert str(log_level).lower() in LOG_LEVEL_DICT, 'only support {}'.format(LOG_LEVEL_DICT.keys())
        Logger.logger.setLevel(LOG_LEVEL_DICT[str(log_level).lower()])
        if is_color:
            disp_formatter = _ColorfulFormatter(log_format)
        else:
            disp_formatter = logging.Formatter(log_format)
        save_formatter = logging.Formatter(log_format)

        if is_disp_on_terminal:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(DEFAULT_LOG_LEVEL)
            ch.setFormatter(disp_formatter)
            Logger.logger.addHandler(ch)

        if save_dir is not None:
            FileHelper.make_dirs(save_dir)
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(DEFAULT_LOG_LEVEL)
            fh.setFormatter(save_formatter)
            Logger.logger.addHandler(fh)

    @staticmethod
    def get_logger():
        return Logger


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
