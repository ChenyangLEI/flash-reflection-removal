
class Registry(dict):
    def register(self, cls):
        self[cls.__name__] = cls
        return cls

    def register_as(self, name):
        def _register(cls):
            self[name] = cls
            return cls
        return _register
