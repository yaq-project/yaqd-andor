__all__ = ["AndorSdk2Ixon"]

import asyncio
import numpy as np

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List, Union
from . import _andor_sdk2
import ctypes
from time import sleep

class AndorSdk2Ixon(_andor_sdk2.AndorSDK2):
    _kind = "andorsdk2-ixon"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._channel_names = ["image"]
        self._channel_units = {"image": "counts"}
        self.stop_update==True
        # find devices
        code, device_count = self.sdk.GetAvailableCameras()
        if device_count == 0:
            raise ConnectionError("No devices found.")
        if code != int(20002):
            self.logger.debug(f"getavailablecameras error {str(self.errorlookup(code))}")

        for i in range(device_count):
            ret,serial = self.sdk.GetCameraSerialNumber()
            if int(serial) == int(self._config["serial_number"]):
                print("    Serial No   : ", serial)
                self.hndl=self.sdk.GetCameraHandle(i)
                break
        else:
            raise ConnectionError(
                r"device with serial number {0} not found".format(self._config["serial_number"])
            )

        #self.sensor_info = {}
        #self.logger.debug(self.sensor_info)
        self.sensor_width = self._config["sensor_width"]
        self.sensor_height = self._config["sensor_height"]
        self.max_width = self._config["pixel_width"]
        self.max_height = self._config["pixel_height"]
        self.preamp_gain=int(self._config["simple_preamp_gain_control"])
        self.exposure_time=float(self._config["exposure_time"])
        self.has_mono=bool(self._config["has_monochromator"])
        self._set_aoi()
        self._gen_mappings()

        # implement config, state features
        # see p97 of SDK2 on the conversion of noise filters from sdk3 to sdk2 or v-v.
        # Commented out until further work done to establish differences between
        # these filters and SDK3.  These may be hardcoded here rather than in config.
        #self.sdk.Filter_SetMode(int(self._config["spurious_noise_filter"]))
        #self.sdk.Filter_SetThreshold(float(self._config["filter_threshold"]))
        self.sdk.SetPreAmpGain(self.preamp_gain)
        self.logger.info(
                f"PreAmpGain: {self.preamp_gain}"
            )

        self.sdk.SetExposureTime(self.exposure_time)
        self.logger.info(
                f"Exposure Time: {self.exposure_time} sec"
            )
        # aoi currently in config, so only need to run on startup

        self._set_temperature()
        self.sdk.SetShutter(int(0),int(1),int(100),int(100))
        self.stop_update=False

    def _gen_mappings(self):
        """Get map."""
        if self.has_mono:
            # translate inputs into appropriate internal units
            spec_inclusion_angle_rad = np.radians(float(self._config["inclusion_angle"]))
            spec_focal_length_tilt_rad = np.radians(float(self._config["focal_length_tilt"]))
            pixel_width_mm =float(self.sensor_width/self.max_width)/1000.00
            self._channel_mappings = {"image": ["wavelengths", "y_index"]}
            self._mapping_units = {"wavelengths": "nm", "y_index": "None"}
            self.spec_position = self._config["spectrometer_position"]

            # create array
            i_pixel = np.array(self.x_ai, dtype=float)
            eff_pixel_width_mm=float(int(self.binning)*pixel_width_mm)
            # calculate terms
            x = np.arcsin(
                (1e-6 * self._config["order"] * self._config["grooves_per_mm"] * self.spec_position)
                / (2 * np.cos(spec_inclusion_angle_rad / 2.0))
            )
            A = np.sin(x - spec_inclusion_angle_rad / 2)
            B = np.sin(
                (spec_inclusion_angle_rad)
                + x
                - (spec_inclusion_angle_rad / 2)
                - np.arctan(
                    (
                        eff_pixel_width_mm * (i_pixel - self._config["calibration_pixel"])
                        + self._config["focal_length"] * spec_focal_length_tilt_rad
                    )
                    / (self._config["focal_length"] * np.cos(spec_focal_length_tilt_rad))
                )
            )
            out = ((A + B) * float(1e6)) / (float(self._config["order"]) * float(self._config["grooves_per_mm"]))
            self._mappings= {"x_index": out, "y_index": self.y_ai}
        else:
            self._channel_mappings = {"image": ["x_index", "y_index"]}
            self._mapping_units = {"x_index": "None", "y_index": "None"}
            self._mappings = {"x_index": self.x_ai, "y_index": self.y_ai}
        return out


    def _set_aoi(self):
        aoi_keys = ["aoi_binning", "aoi_width", "aoi_left", "aoi_height", "aoi_top"]
        binning, width, left, height, top = [self._config[k] for k in aoi_keys]
        left=int(left)
        width=int(width)
        height=int(height)
        top=int(top)
        binning = int(binning)  # equal xy binning, so only need 1 index

        # handle defaults (maximum sizes)
        if left is None:
            left = 1
        if top is None:
            top = 1
        if width is None:
            width = self.max_width - left + 1
        if height is None:
            height = self.max_height - top + 1

        arrwidth = int(int(width)/int(binning))
        arrheight = int(int(height)/int(binning))

        right=left+arrwidth-1
        bottom=top+arrheight-1

        self.logger.debug(f"{self.max_width}, {self.max_height}, {binning}, {width}, {height}, {top}")
        w_extent = int(width * binning + (left - 1))
        h_extent = int(height * binning + (top - 1))
        if w_extent > self.max_width:
            raise ValueError(f"height extends over {w_extent} pixels, max is {self.max_width}")
        if h_extent > self.max_height:
            raise ValueError(f"height extends over {h_extent} pixels, max is {self.max_height}")

        code= self.sdk.SetImage(int(binning),int(binning),int(left),int(right),int(top),int(bottom))
        self.logger.info(f"binning={binning}, left_aoi=pixel {left}, right_aoi=pixel {right}, top_aoi=pixel {top}, bottom_aoi=pixel {bottom}.")
        if code != 20002:
            raise ValueError(str(self.errorlookup(code)))
            return code
        else:
            self.buffer_size=int(int(width)*int(height))
            self.buffer=np.zeros([arrwidth, arrheight], dtype=int)
            self.buffer=np.ascontiguousarray(self.buffer, dtype=int)

            # apply shape, mapping
            self._channel_shapes = {
                "image": (int(height), int(width))
            }
            self.x_ai = np.arange(int(left), int(left + width * binning), int(binning), dtype=int)[None, :]
            self.y_ai = np.arange(int(top), int(top + height * binning), int(binning), dtype=int)[:, None]
            self.binning= int(binning)
            return 0


    def _set_temperature(self):
        self.sensor_cooling = self._config["sensor_cooling"]
        if self.sensor_cooling:
            self.sensor_temp_control = int(self._config["sensor_temperature"])
            code=self.sdk.CoolerON()
            if code != 20002:
                raise ValueError(str(self.errorlookup(code)))
            code=self.sdk.SetTemperature(self.sensor_temp_control)
            if code != 20002:
                raise ValueError(str(self.errorlookup(code)))
            self.logger.info(f"Sensor is cooling.  Target temp is {self.sensor_temp_control} C.")
            self._loop.run_in_executor(None, self._check_temp_stabilized)
        else:
            self.sensor_temp = float(self.sdk.GetTemperature())
            self.logger.info(f"Sensor is not cooled.  Current temp is {self.sensor_temp} C.")


    def _check_temp_stabilized(self):
        code,self.sensor_temp = self.sdk.GetTemperature()
        diff = float(self.sensor_temp_control) - float(self.sensor_temp)
        while np.abs(diff) > 1.0:  #this is a tolerance, and is subject to change
            self.logger.info(
                f"Sensor is cooling.  Target: {self.sensor_temp_control} C.  Current: {self.sensor_temp:0.2f} C."
            )
            sleep(3)
            code,self.sensor_temp = self.sdk.GetTemperature()
            diff = float(self.sensor_temp) - float(self.sensor_temp_control)

        self.logger.info("Sensor temp is stabilized.")


    def set_spec_position(self, spec_position):
        # Sets the CCD mapping to a new spectrometer position (nm)
        if self.has_mono:
            self.spec_position=float(spec_position)
            self._gen_mappings()
            self.logger.info(f"new spectrometer position: {self.spec_position} nm")
        else:
            self.logger.info(f"Attempt to change spectrometer position when monochromator portion not enabled.")


    def get_spec_position(self):
        # Gets the CCD mapping to the spectrometer position (nm)
        if self.has_mono:
            return self.spec_position
        else:
            self.logger.info(f"Attempt to change spectrometer position when monochromator portion not enabled.")
            return float(0.00)


    def set_exposure_time(self, exposure_time):
        # Sets the exposure time in seconds (float)
        self.stop_update=True
        while self._busy==True:
            sleep(0.10)
        code=self.sdk.SetExposureTime(float(exposure_time))
        if code != 20002:
            raise ValueError(str(self.errorlookup(code)))
        else:
            self.exposure_time=exposure_time
            self.logger.info(f"New exposure time is {self.exposure_time} sec.")
        self.stop_update=False
        return code


    def get_exposure_time(self):
        # Gets the exposure time in seconds (float)
        return self.exposure_time


    def get_spec_position(self):
        # Gets the CCD mapping to the spectrometer position (nm)
        if self.has_mono:
            return self.spec_position
        else:
            self.logger.info(f"Attempt to change spectrometer position when monochromator portion not enabled.")
            return float(0.00)


    def close(self):
        #stop loop
        self.stop_update=True
        while self._busy==True:
            sleep(0.10)
        self.sdk.SetShutter(int(0),int(2),int(100),int(100))
        sleep(0.50)
        self.sdk.CoolerOFF()
        self.sensor_temp_control=int(25.0)
        code=self.sdk.SetTemperature(self.sensor_temp_control)
        sleep(0.20)
        code,self.sensor_temp = self.sdk.GetTemperature()
        diff = float(self.sensor_temp_control) - float(self.sensor_temp)
        while np.abs(diff) > 6.0:  #this is a tolerance, and is subject to change
            self.logger.info(
                f"Sensor is warming to RT.  Target: {self.sensor_temp_control} C.  Current: {self.sensor_temp:0.2f} C."
            )
            sleep(3)
            code,self.sensor_temp = self.sdk.GetTemperature()
            diff = float(self.sensor_temp_control) - float(self.sensor_temp)

        sleep(0.2)
        code=self.sdk.ShutDown()
        if code != 20002:
            raise ValueError(str(self.errorlookup(code))+ ", not closed properly" )
        else:
            self.logger.info(
                f"Sensor warmed to RT. Camera closed."
            )
        return