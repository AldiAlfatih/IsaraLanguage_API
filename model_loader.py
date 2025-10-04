import os
from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model as keras_load_model


class PatchedInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        # Some legacy H5 models store 'batch_shape' instead of 'batch_input_shape'.
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)


def load_bisindo_model(path: str):
    """Robust loader for legacy H5 that may contain 'batch_shape' in InputLayer config.

    Tries loading with a patched InputLayer via custom_objects.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    try:
        model = keras_load_model(
            path,
            compile=False,
            custom_objects={
                "InputLayer": PatchedInputLayer,
                # Some legacy models serialize mixed precision policy as 'DTypePolicy'.
                # Map it to the modern tf.keras.mixed_precision.Policy for compatibility.
                "DTypePolicy": tf.keras.mixed_precision.Policy,
            },
        )
        return model
    except TypeError as e:
        # Re-raise with more context
        raise TypeError(
            f"Failed to load model '{path}' with patched InputLayer. "
            f"Original error: {e}"
        )
