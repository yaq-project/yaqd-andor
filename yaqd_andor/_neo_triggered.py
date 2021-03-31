__all__ = ["NeoTriggered"]

import asyncio
from re import L
import numpy as np

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List
from . import atcore 
from . import features

import os
os.chdir(os.path.dirname(__file__))

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class NeoTriggered(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    _kind = "neo-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._channel_names = ["image"]
        self._channel_shapes = {"image": (1920, 1080)}
        self.sdk3 = ATCore() # Initialise SDK3
        # find devices
        self.is_virtual = self._config["is_virtual"]
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

        if not self.is_virtual:
            self.sdk3.set_enumerated_string(
                self.hndl,
                "SimplePreAmpGainControl",
                "16-bit (low noise & high well capacity)"
            )
            # self.sdk3.set_bool(self.hndl, "MetadataEnable", True)

        if False:
            # set trigger mode
            self.sdk3.set_enumerated_string(
                self.hndl, "TriggerMode", "Advanced" if self._config["is_virtual"] else "Internal"
            )
            # set exposure time
            print("exposure time is implemented:", bool(self.sdk3.is_implemented(self.hndl, "ExposureTime")))
            print("exposure time is read only:", bool(self.sdk3.is_readonly(self.hndl, "ExposureTime")))
            import time
            while True:
                if self.sdk3.is_writable(self.hndl, "ExposureTime"):
                    print("writable!")
                    break
                else:
                    print("not writable")
                    time.sleep(1)
            print("exposure time is writable:", bool(self.sdk3.is_writable(self.hndl, "ExposureTime")))
            self.sdk3.set_float(self.hndl, "ExposureTime", self._state["exposure_time"])
        # set cycle mode to fixed
        # self.sdk3.set_enumerated_string(
        #     self.hndl, "CycleMode", "Fixed"
        # )
        print(self.sdk3.get_enumerated_string(
            self.hndl, "SimplePreAmpGainControl"
        ))


    async def _measure(self):
        # queue buffer
        # todo: queue buffer for n frames
        imageSizeBytes = self.sdk3.get_int(self.hndl, "ImageSizeBytes")
        # acquire frame
        buf = np.empty((imageSizeBytes,), dtype='B')
        try:
            self.sdk3.queue_buffer(self.hndl, buf.ctypes.data, imageSizeBytes)
            self.sdk3.command(self.hndl, "AcquisitionStart")
            self.logger.debug("Waiting on buffer")
            (returnedBuf, returnedSize) = self.sdk3.wait_buffer(self.hndl)  # hangs
            self.logger.debug("Done waiting on buffer")
        except ATCoreException as err:
            self.logger.error(f"SDK3 Error {err}")

        class ArrayInterface:
            def __init__(self, buf, shape, strides):
                self.__array_interface__ = {
                    "shape": shape,
                    "typestr": "<2u",
                    "data": buf,
                    "strides": strides,
                    "version": 3,
                }
        stride = self.sdk3.get_int(self.hndl, "AOIStride")
        pixels = np.array(ArrayInterface(buf.data, self._channel_shapes["image"], (stride, 1)))
        self.logger.debug(f"{pixels.size}, {np.prod(self._channel_shapes['image'])}")
        self.sdk3.command(self.hndl,"AcquisitionStop")
        self.sdk3.flush(self.hndl)
        pixels = np.ascontiguousarray(pixels)
        print(pixels.__array_interface__)
        return {"image": np.ascontiguousarray(pixels)}
        
 
    def get_sensor_info(self):
        pass

    def interrupt(self):
        """stop measure before exposure time is reached.
        """
        pass

    def close(self):
        self.sdk3.close(self.hndl)

    def _get_feature(self, feature):
        call = type_to_call["int"].format("get")
        if self.sdk3.is_implemented(self.hndl, call):
            if self.sdk3.is_readable(self.hndl, call):
                return self.sdk3.__getattribute__(call)(self.hndl, feature.name)
            else:
                raise TypeError(f"call {call} is not readable")
        else:
            raise ValueError(f"call {call} is not implemented")

    def _set_feature(self, feature):
        call = type_to_call["int"].format("set")
        if self.sdk3.is_readonly(self.hndl, call):
            return TypeError(f"Cannot write.  Feature {call} is read only")
        if self.sdk3.is_implemented(self.hndl, call):
            while not self.sdk3.is_writable(self.hndl, call):
                return self.sdk3.__getattribute__(call)(self.hndl, feature.name, feature.value)
        else:
            raise ValueError(f"call {call} is not implemented")
