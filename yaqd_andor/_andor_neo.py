__all__ = ["AndorNeo"]

import asyncio
import numpy as np
from time import sleep

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List, Union
from . import atcore 
from . import features

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class AndorNeo(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    _kind = "andor-neo"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._channel_names = ["image"]
        self.sdk3 = ATCore() # Initialise SDK3
        # find devices
        device_count = self.sdk3.get_int(self.sdk3.AT_HNDL_SYSTEM, "DeviceCount")
        if device_count == 0:
            raise ConnectionError("No devices found.")
        # select device
        for i in range(device_count):
            temp = self.sdk3.open(i)
            serial = self.sdk3.get_string(temp, "SerialNumber")
            if serial == self._config["serial"]:
                self.hndl = temp
                print("    Serial No   : ", serial)
                break
            self.sdk3.close(temp)
        else:
            raise ConnectionError(
                r"device with serial number {0} not found".format(self._config["serial"])
            )

        self.features = {}
        for k, v in features.specs.items():
            if "n" in v.availability:
                try:
                    self.features[k] = features.obj_from_spec(self.sdk3, self.hndl, v)
                except NotImplementedError:
                    self.logger.debug(
                        f"feature {v.sdk_name} is supposed to be implemented, but is not!"
                    )
                    pass
                else:
                    self.logger.debug(
                        f"{k}, {self.features[k].is_implemented}, {self.features[k].is_readonly}"
                    )

        # only need to poll once
        self.sensor_info = {}
        for k in [
                "sensor_width", "sensor_height", "pixel_height", "pixel_width"
            ]:
            try:
                self.sensor_info[k] = self.features[k].get()
            except ATCoreException as err:
                pass
        self.logger.debug(self.sensor_info)

        # implement config, state features
        self.features["spurious_noise_filter"].set(self._config["spurious_noise_filter"])
        self.features["static_blemish_correction"].set(self._config["static_blemish_correction"])
        self.features["electronic_shuttering_mode"].set(self._config["electronic_shuttering_mode"])
        self.features["simple_preamp_gain_control"].set(self._config["simple_preamp_gain_control"])
        self.features["exposure_time"].set(self._state["exposure_time"])
        # aoi currently in config, so only need to run on startup
        self._set_aoi()
        self._set_temperature()

    def _set_aoi(self):
        aoi_keys = ["aoi_binning", "aoi_width", "aoi_left", "aoi_height", "aoi_top"]
        binning, width, left, height, top = [
            self._config[k] for k in aoi_keys
        ]
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
        w_extent = width * binning + (left-1)
        h_extent = height * binning  + (top-1)
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
        x_ai = np.arange(left, width * binning, binning)[None, :]
        y_ai = np.arange(top, height * binning, binning)[:, None]
        
        x_index = x_ai.__array_interface__
        x_index["data"] = x_ai.tobytes()
        y_index = y_ai.__array_interface__
        y_index["data"] = y_ai.tobytes()
        
        self._mappings = {
            "x_index": x_index,
            "y_index": y_index
        }

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
        while abs(diff) > 1.:
            self.logger.info(
                f"Sensor is cooling.  Target: {set_temp} C.  Current: {sensor_temp:0.2f} C."
            )
            sleep(5)
            set_temp = self.features["temperature_control"].get()
            sensor_temp = self.features["sensor_temperature"].get()
            diff = float(set_temp) - sensor_temp
        self.logger.info("Sensor temp is stabilized.")

    async def _measure(self):
        image_size_bytes = self.features["image_size_bytes"].get()
        buf = np.empty((image_size_bytes,), dtype='B')
        try:
            self.sdk3.queue_buffer(self.hndl, buf.ctypes.data, image_size_bytes)
            # acquire frame
            self.features["acquisition_start"]()
            self.logger.debug("Waiting on buffer")
            (returnedBuf, returnedSize) = await self._loop.run_in_executor(
                None, self.sdk3.wait_buffer, self.hndl
            )
            self.logger.debug("Done waiting on buffer")
            self.features["acquisition_stop"]()
        except ATCoreException as err:
            self.logger.error(f"SDK3 Error {err}")

        class ArrayInterface:
            def __init__(self, buf, shape, strides):
                self.__array_interface__ = {
                    "shape": shape,
                    "typestr": "<u2",
                    "data": buf,
                    "strides": strides,
                    "version": 3,
                }
        stride = self.features["aoi_stride"].get()
        pixels = np.array(ArrayInterface(buf.data, self._channel_shapes["image"], (stride, 2)))
        self.logger.debug(f"{pixels.size}, {np.prod(self._channel_shapes['image'])}")
        pixels = np.ascontiguousarray(pixels)
        arrayinterface = pixels.__array_interface__
        arrayinterface["data"] = pixels.tobytes()
        self.sdk3.flush(self.hndl)

        return {"image": arrayinterface}

    def get_sensor_info(self):
        return self.sensor_info

    def get_feature_names(self) -> List[str]:
        return [v.sdk_name for v in self.features.values()]

    def get_feature_value(self, k:str) -> Union(int, bool, float, str):
        feature = self.features[k]
        return feature.get()

    def get_feature_options(self, k:str) -> List[str]:
        feature = self.features[k]
        # if isinstance(feature, features.SDKEnum):
        return feature.options()
        # else:
        #     raise ValueError(f"feature {feature} is of type {type(feature)}, not `SDKEnum`.")

    def close(self):
        self.sdk3.close(self.hndl)
