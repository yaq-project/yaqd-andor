__all__ = ["AndorNeo"]

import asyncio
import numpy as np
import os

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List
from . import atcore
from . import features

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class AndorSona(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    _kind = "andor-neo"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._channel_names = ["image"]
        initial_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(__file__))
            self.sdk = ATCore()  # Initialise SDK3
        finally:
            os.chdir(initial_cwd)
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

        self.sdk3.set_float(self.hndl, "ExposureTime", self._state["exposure_time"])

        height = self.sdk3.get_int(self.hndl, "SensorHeight")
        width = self.sdk3.get_int(self.hndl, "SensorWidth")
        self._channel_shapes = {"image": (height, width)}

    async def _measure(self):
        imageSizeBytes = self.sdk3.get_int(self.hndl, "ImageSizeBytes")
        buf = np.empty((imageSizeBytes,), dtype="B")
        try:
            self.sdk3.queue_buffer(self.hndl, buf.ctypes.data, imageSizeBytes)
            # acquire frame
            self.sdk3.command(self.hndl, "AcquisitionStart")
            self.logger.debug("Waiting on buffer")
            (returnedBuf, returnedSize) = self.sdk3.wait_buffer(self.hndl)
            self.logger.debug("Done waiting on buffer")
            self.logger.debug(f"{imageSizeBytes}, {returnedSize}")
            self.sdk3.command(self.hndl, "AcquisitionStop")
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
        pass

    def close(self):
        self.sdk3.close(self.hndl)
