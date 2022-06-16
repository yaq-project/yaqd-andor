__all__ = ["AndorSimcam"]

import asyncio
import numpy as np

from typing import Dict, Any, List
from . import atcore
from . import features
from . import _andor_sdk3

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class AndorSimcam(_andor_sdk3.AndorSDK3):
    _kind = "andor-simcam"
    # state_features also have avro messages for setting, reading, options, etc.
    state_features = ["exposure_time", "pixel_readout_rate", "electronic_shuttering_mode"]

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._set_aoi()

    def _set_aoi(self):
        aoi_keys = ["aoi_hbin", "aoi_vbin", "aoi_width", "aoi_left", "aoi_height", "aoi_top"]
        aoi_hbin, aoi_vbin, width, left, height, top = [self._config[k] for k in aoi_keys]

        # check if aoi is within sensor limits
        max_width = self.features["sensor_width"].get()
        max_height = self.features["sensor_height"].get()

        # handle defaults (maximum sizes)
        if left is None:
            left = 1
        if top is None:
            top = 1
        if width is None:
            width = max_width - left + 1
        if height is None:
            height = max_height - top + 1
        width //= aoi_hbin
        height //= aoi_vbin

        self.logger.debug(
            f"{max_width}, {max_height}, {aoi_hbin}, {aoi_vbin}, {width}, {height}, {top}"
        )
        w_extent = width * aoi_hbin + (left - 1)
        h_extent = height * aoi_vbin + (top - 1)
        if w_extent > max_width:
            raise ValueError(f"height extends over {w_extent} pixels, max is {max_width}")
        if h_extent > max_height:
            raise ValueError(f"height extends over {h_extent} pixels, max is {max_height}")

        for name, val in zip(
            ["hbin", "vbin", "width", "left", "height", "top"],
            [aoi_hbin, aoi_vbin, width, left, height, top],
        ):
            try:  # none are writable, for some reason
                self.features[f"aoi_{name}"].set(val)
            except Exception as e:
                self.logger.error(f"set_aoi: {e}")

        # apply shape, mapping
        self._channel_shapes = {
            "image": (self.features["aoi_height"].get(), self.features["aoi_width"].get())
        }
        x_ai = np.arange(left, left + width * aoi_hbin, aoi_hbin)[None, :]
        y_ai = np.arange(top, top + height * aoi_vbin, aoi_vbin)[:, None]

        x_index = x_ai.__array_interface__
        x_index["data"] = x_ai.tobytes()
        y_index = y_ai.__array_interface__
        y_index["data"] = y_ai.tobytes()

        self._mappings = {"x_index": x_index, "y_index": y_index}

        for k in ["aoi_height", "aoi_width", "aoi_top", "aoi_left", "aoi_hbin", "aoi_vbin"]:
            self.logger.debug(f"{k}: {self.features[k].get()}")
