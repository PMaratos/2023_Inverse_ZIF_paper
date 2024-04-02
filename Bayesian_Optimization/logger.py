import logging

class Logger(logging.Logger):

    def __init__(self, name, level=logging.NOTSET, fileName=None, formatting="%(asctime)s - %(levelname)s - %(message)s"):
        
        super().__init__(name, level)
        self.extra_info = None

        if fileName is not None:
            handler = logging.FileHandler(filename=fileName)            
            handler.setFormatter(logging.Formatter(formatting))
            self.addHandler(handler)

    def createLogMessage(self, prefix, message):
        return "[" + prefix + "] " + message

    def debug(self, prefix, message, *args, **kwargs):
        super().debug(self.createLogMessage(prefix,message), *args, **kwargs)

    def info(self, prefix, message, *args, **kwargs):
        super().info(self.createLogMessage(prefix,message), *args, **kwargs)

    def warn(self, prefix, message, *args, **kwargs):
        super().warn(self.createLogMessage(prefix,message), *args, **kwargs)

    def error(self, prefix, message, *args, **kwargs):
        super().error(self.createLogMessage(prefix,message), *args, **kwargs)

    def fatal(self, prefix, message, *args, **kwargs):
        super().critical(self.createLogMessage(prefix,message), *args, **kwargs)