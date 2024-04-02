import logging

class Logger(logging.Logger):

    def __init__(self, name, level=logging.NOTSET, fileName=None, formatting="%(asctime)s - %(levelname)s - %(message)s"):
        
        super().__init__(name, level)
        self.extra_info = None

        if fileName is not None:
            handler = logging.FileHandler(filename=fileName)            
            handler.setFormatter(logging.Formatter(formatting))
            self.addHandler(handler)

    def debug(self, message, *args, **kwargs):
        super().debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        super().info(message, *args, **kwargs)

    def warn(self, message, *args, **kwargs):
        super().warn(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        super().error(message, *args, **kwargs)

    def fatal(self, message, *args, **kwargs):
        super().critical(message, *args, **kwargs)