import datetime
import warnings

from .state import State


class StatefulBaseClass:
    def setup(self, state=State()):
        return state

    def show_config(self, registered_objects=None):
        if registered_objects is None:
            registered_objects = []

        config = {}
        for key, value in self.__dict__.items():
            if isinstance(value, StatefulBaseClass) and value not in registered_objects:
                registered_objects.append(value)
                config[str(key)] = value.show_config(registered_objects)
            else:
                config[str(key)] = str(value)
        return config

    def load(self, path):
        warnings.warn(
            "StatefulBaseClass.load is not implemented in the PyTorch port yet. "
            "The original JAX package also keeps save/load commented out.",
            stacklevel=2,
        )
        raise NotImplementedError(f"Loading from {path} is not implemented yet")

    @staticmethod
    def default_checkpoint_path(class_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"./{class_name} {timestamp}.pkl"