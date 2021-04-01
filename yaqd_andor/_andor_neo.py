__all__ = ["AndorNeo"]

import asyncio
import numpy as np

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List
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
                print("    Serial No   : ",serial)
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
                    self.features[k] = features.obj_from_spec(
                        self.sdk3, self.hndl, v
                    )
                except NotImplementedError:
                    self.logger.debug(
                        f"feature {v.sdk_name} is supposed to be implemented, but is not!"
                    )
                    pass
                else:
                    self.logger.debug(f"{k}, {self.features[k].is_implemented}, {self.features[k].is_readonly}")

        # only need to poll once
        self.sensor_info = {}
        for k in [
                "sensor_width", "sensor_height", "pixel_height", "pixel_width"
            ]:
            try:
                self.sensor_info[k] = self.features[k].get()
            except ATCoreException as err:
                # self.logger.log(err.msg)
                pass
        self.logger.debug(self.sensor_info)

        # implement config, state features
        self.features["exposure_time"].set(self._state["exposure_time"])
        self.features["simple_preamp_gain_control"].set(self._config["simple_preamp_gain_control"])

        # aoi currently in config, so only need to run on startup
        self._set_aoi()

        # apply channel shape
        self._channel_shapes = {"image": (self.features["aoi_height"], self.features["aoi_width"])}

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
            width = (max_width - left + 1) // binning
        if height is None:
            height = (max_height - top + 1) // binning

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

        # todo: apply aoi to mapping

        for k in ["aoi_height", "aoi_width", "aoi_top", "aoi_left", "aoi_binning"]:
            self.logger.debug(f"{k}: {self.features[k].get()}")

    async def _measure(self):
        imageSizeBytes = self.sdk3.get_int(self.hndl, "ImageSizeBytes")
        buf = np.empty((imageSizeBytes,), dtype='B')
        try:
            self.sdk3.queue_buffer(self.hndl, buf.ctypes.data, imageSizeBytes)
            # acquire frame
            self.sdk3.command(self.hndl, "AcquisitionStart")
            self.logger.debug("Waiting on buffer")
            (returnedBuf, returnedSize) = self.sdk3.wait_buffer(self.hndl)
            self.logger.debug("Done waiting on buffer")
            self.logger.debug(f"{imageSizeBytes}, {returnedSize}")
            self.sdk3.command(self.hndl,"AcquisitionStop")
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
        stride = self.sdk3.get_int(self.hndl, "AOIStride")
        pixels = np.array(ArrayInterface(buf.data, self._channel_shapes["image"], (stride, 2)))
        self.logger.debug(f"{pixels.size}, {np.prod(self._channel_shapes['image'])}")
        pixels = np.ascontiguousarray(pixels)
        arrayinterface = pixels.__array_interface__
        arrayinterface["data"] = pixels.tobytes()
        self.sdk3.flush(self.hndl)

        return {"image": arrayinterface}

    def get_sensor_info(self):
        return self.sensor_info

    def list_features(self):
        pass

    def close(self):
        self.sdk3.close(self.hndl)