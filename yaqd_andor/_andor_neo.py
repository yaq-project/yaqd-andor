__all__ = ["AndorNeo"]

import asyncio
import numpy as np
from time import sleep

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List, Union
from . import atcore
from . import features
from . import _andor_sdk3

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class AndorNeo(_andor_sdk3.AndorSDK3):
    _kind = "andor-neo"
    state_features = [
        "exposure_time",
        "pixel_readout_rate",
        "electronic_shuttering_mode",
        "simple_preamp_gain_control",
        "spurious_noise_filter",
        "static_blemish_correction",
    ]

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._set_aoi()
        self._set_temperature()

    def _set_aoi(self):
        aoi_keys = ["aoi_binning", "aoi_width", "aoi_left", "aoi_height", "aoi_top"]
        binning, width, left, height, top = [self._config[k] for k in aoi_keys]
        binning = int(binning[0])  # equal xy binning, so only need 1 index

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
        width //= binning
        height //= binning

        self.logger.debug(f"{max_width}, {max_height}, {binning}, {width}, {height}, {top}")
        w_extent = width * binning + (left - 1)
        h_extent = height * binning + (top - 1)
        if w_extent > max_width:
            raise ValueError(f"height extends over {w_extent} pixels, max is {max_width}")
        if h_extent > max_height:
            raise ValueError(f"height extends over {h_extent} pixels, max is {max_height}")

        self.features["aoi_binning"].set(f"{binning}x{binning}")
        self.features["aoi_width"].set(width)
        self.features["aoi_left"].set(left)
        self.features["aoi_height"].set(height)
        self.features["aoi_top"].set(top)

        # apply shape, mapping
        self._channel_shapes = {
            "image": (self.features["aoi_height"].get(), self.features["aoi_width"].get())
        }
        x_ai = np.arange(left, left + width * binning, binning)[None, :]
        y_ai = np.arange(top, top + height * binning, binning)[:, None]

        x_index = x_ai.__array_interface__
        x_index["data"] = x_ai.tobytes()
        y_index = y_ai.__array_interface__
        y_index["data"] = y_ai.tobytes()

        self._mappings = {"x_index": x_index, "y_index": y_index}

        for k in ["aoi_height", "aoi_width", "aoi_top", "aoi_left", "aoi_binning"]:
            self.logger.debug(f"{k}: {self.features[k].get()}")

    def _set_temperature(self):
        # possible_temps = self.features["temperature_control"].options()
        sensor_cooling = self._config["sensor_cooling"]
        self.features["sensor_cooling"].set(sensor_cooling)
        if sensor_cooling:
            set_temp = self.features["temperature_control"].get()
            self.logger.info(f"Sensor is cooling.  Target temp is {set_temp} C.")
            self._loop.run_in_executor(None, self._check_temp_stabilized)
        else:
            sensor_temp = self.features["sensor_temperature"].get()
            self.logger.info(f"Sensor is not cooled.  Current temp is {sensor_temp} C.")

        status = self.features["temperature_status"].get()

    def _check_temp_stabilized(self):
        set_temp = self.features["temperature_control"].get()
        sensor_temp = self.features["sensor_temperature"].get()
        diff = float(set_temp) - sensor_temp
        while abs(diff) > 1.0:
            self.logger.info(
                f"Sensor is cooling.  Target: {set_temp} C.  Current: {sensor_temp:0.2f} C."
            )
            sleep(5)
            set_temp = self.features["temperature_control"].get()
            sensor_temp = self.features["sensor_temperature"].get()
            diff = float(set_temp) - sensor_temp
        self.logger.info("Sensor temp is stabilized.")
