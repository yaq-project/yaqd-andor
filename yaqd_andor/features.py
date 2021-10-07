from collections import namedtuple

FeatureSpec = namedtuple("FeatureSpec", ["sdk_name", "type", "availability"])

# adapted from sdk3 manual
# note some table entries were omitted (focused on neo, simcam features)
# a = apogee, n = neo, s = simcam, z = zyla
specs = dict(
    acquisition_start=FeatureSpec("AcquisitionStart", "command", "nsz"),
    acquisition_stop=FeatureSpec("AcquisitionStop", "command", "nsz"),
    aoi_binning=FeatureSpec("AOIBinning", "enumerated", "nz"),
    aoi_hbin=FeatureSpec("AOIHBin", "int", "asz"),
    aoi_height=FeatureSpec("AOIHeight", "int", "ansz"),
    aoi_layout=FeatureSpec("AOILayout", "enumerated", "az"),
    aoi_left=FeatureSpec("AOILeft", "int", "ansz"),
    aoi_stride=FeatureSpec("AOIStride", "int", "nsz"),  # not in docs, but implemented for simcam
    aoi_top=FeatureSpec("AOITop", "int", "ansz"),
    aoi_vbin=FeatureSpec("AOIVBin", "int", "asz"),
    aoi_width=FeatureSpec("AOIWidth", "int", "ansz"),
    baseline=FeatureSpec("Baseline", "int", "nz"),
    bit_depth=FeatureSpec("BitDepth", "enumerated", "anz"),
    buffer_overflow_event=FeatureSpec("BufferOverflowEvent", "int", "nz"),
    bytes_per_pixel=FeatureSpec("BytesPerPixel", "float", "nz"),
    camera_acquiring=FeatureSpec("CameraAcquiring", "bool", "nsz"),
    camera_dump=FeatureSpec("CameraDump", "command", "nz"),
    camera_family=FeatureSpec("CameraFamily", "string", "a"),
    camera_memory=FeatureSpec("CameraMemory", "int", "a"),
    camera_model=FeatureSpec("CameraModel", "string", "nsz"),
    camera_name=FeatureSpec("CameraName", "string", "anz"),
    cycle_mode=FeatureSpec("CycleMode", "enumerated", "nsz"),
    electronic_shuttering_mode=FeatureSpec("ElectronicShutteringMode", "enumerated", "nsz"),
    event_enable=FeatureSpec("EventEnable", "bool", "nz"),
    events_missed_event=FeatureSpec("EventsMissedEvent", "int", "nz"),
    event_selector=FeatureSpec("EventSelector", "enumerated", "nz"),
    exposure_time=FeatureSpec("ExposureTime", "float", "nsz"),
    exposure_end_event=FeatureSpec("ExposureEndEvent", "int", "nz"),
    exposure_start_event=FeatureSpec("ExposureStartEvent", "int", "nz"),
    external_trigger_delay=FeatureSpec("ExposureTriggerDelay", "float", "z"),
    fan_speed=FeatureSpec("FanSpeed", "enumerated", "ansz"),
    fast_aoi_frame_rate_enable=FeatureSpec("FastAOIFrameRateEnable", "bool", "nz"),
    firmware_version=FeatureSpec("FirmwareVersion", "string", "anz"),
    frame_count=FeatureSpec("FrameCount", "int", "ansz"),
    frame_rate=FeatureSpec("FrameInterval", "float", "ansz"),
    full_aoi_control=FeatureSpec("FullAOIControl", "bool", "nz"),
    image_size_bytes=FeatureSpec("ImageSizeBytes", "int", "nsz"),
    interface_type=FeatureSpec("InterfaceType", "string", "anz"),
    max_interface_transfer_rate=FeatureSpec("MaxInterfaceTransferRate", "float", "nz"),
    metadata_enable=FeatureSpec("MetadataEnable", "bool", "nz"),
    metadata_frame=FeatureSpec("MetadataFrame", "bool", "nz"),
    metadata_timestamp=FeatureSpec("MetadataTimestamp", "bool", "nz"),
    overlap=FeatureSpec("Overlap", "bool", "anz"),
    pixel_correction=FeatureSpec("PixelCorrection", "enumerated", "s"),
    pixel_encoding=FeatureSpec("PixelEncoding", "enumerated", "nsz"),
    pixel_height=FeatureSpec("PixelHeight", "enumerated", "nsz"),
    pixel_readout_rate=FeatureSpec("PixelReadoutRate", "enumerated", "ansz"),
    pixel_width=FeatureSpec("PixelWidth", "float", "ansz"),
    readout_time=FeatureSpec("ReadoutTime", "float", "nz"),
    sensor_cooling=FeatureSpec("SensorCooling", "bool", "ansz"),
    sensor_height=FeatureSpec("SensorHeight", "int", "ansz"),
    sensor_readout_mode=FeatureSpec("SensorReadoutMode", "enumerated", "z"),
    sensor_temperature=FeatureSpec("SensorTemperature", "float", "ansz"),
    sensor_width=FeatureSpec("SensorWidth", "int", "ansz"),
    serial_number=FeatureSpec("SerialNumber", "string", "snz"),
    simple_preamp_gain_control=FeatureSpec("SimplePreampGainControl", "enumerated", "nz"),
    software_trigger=FeatureSpec("SoftwareTrigger", "command", "nz"),
    spurious_noise_filter=FeatureSpec("SpuriousNoiseFilter", "bool", "nz"),
    static_blemish_correction=FeatureSpec("StaticBlemishCorrection", "bool", "nz"),
    temperature_control=FeatureSpec("TemperatureControl", "enumerated", "n"),
    temperature_status=FeatureSpec("TemperatureStatus", "enumerated", "anz"),
    timestamp_clock=FeatureSpec("TimestampClock", "int", "nz"),
    timestamp_clock_frequency=FeatureSpec("TimestampClockFrequency", "int", "nz"),
    timestamp_clock_reset=FeatureSpec("TimestampClockReset", "command", "nz"),
    trigger_mode=FeatureSpec("TriggerMode", "enumerated", "ansz"),
    vertically_center_aoi=FeatureSpec("VerticallyCentreAOI", "bool", "nz"),
)


class Feature:
    def __init__(self, sdk, hndl, spec):
        """
        command:  feature()
        get: feature.get()
        set: feature.set(value)
        """
        self.type = spec.type
        self.sdk_name = spec.sdk_name
        self.sdk = sdk
        self.hndl = hndl
        # feature implemented?
        self.is_implemented = bool(self.sdk.is_implemented(self.hndl, self.sdk_name))
        if self.is_implemented:
            self.get = self._get
        else:
            # DDK: not implemented = never available?
            raise NotImplementedError(f"feature {self.sdk_name} is not implemented")
        # is read only?
        self.is_readonly = bool(self.sdk.is_readonly(self.hndl, self.sdk_name))
        if not self.is_readonly:
            self._set_call = f"set_{self.type}"
            self.set = self._set
        self._get_call = f"get_{self.type}"

    def _get(self):
        call = self._get_call
        if self.sdk.is_readable(self.hndl, self.sdk_name):
            return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)
        else:
            raise TypeError(f"{self.sdk_name} call {call} is not readable")

    def _set(self, value):
        call = self._set_call
        if self.sdk.is_writable(self.hndl, self.sdk_name):
            return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name, value)
        else:
            raise ValueError(f"{self.sdk_name} call {call} is not currently writable")


class SDKString(Feature):
    def __init__(self, *args):
        super().__init__(*args)


class SDKFloat(Feature):
    def __init__(self, *args):
        super().__init__(*args)

    def max(self) -> float:
        call = self._get_call + "_max"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)

    def min(self) -> float:
        call = self._get_call + "_min"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)


class SDKInt(Feature):
    def __init__(self, *args):
        super().__init__(*args)

    def max(self) -> int:
        call = self._get_call + "_max"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)

    def min(self) -> int:
        call = self._get_call + "_min"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)


class SDKBool(Feature):
    def __init__(self, *args):
        super().__init__(*args)


class SDKCommand(Feature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        return self.sdk.command(self.hndl, self.sdk_name)


class SDKEnum(Feature):
    def __init__(self, *args):
        super().__init__(*args)
        self._get_call = self._get_call + "_string"
        if not self.is_readonly:
            self._set_call = self._set_call + "_string"

    def options(self):
        """query available feature string options"""
        return self.sdk.get_enumerated_string_options(self.hndl, self.sdk_name)


def obj_from_spec(sdk, hndl, spec):
    if spec.type == "command":
        return SDKCommand(sdk, hndl, spec)
    elif spec.type == "bool":
        return SDKBool(sdk, hndl, spec)
    elif spec.type == "enumerated":
        return SDKEnum(sdk, hndl, spec)
    elif spec.type == "float":
        return SDKFloat(sdk, hndl, spec)
    elif spec.type == "int":
        return SDKInt(sdk, hndl, spec)
    elif spec.type == "string":
        return SDKString(sdk, hndl, spec)
    else:
        print(f"failed to find valid type for {spec}.")
        pass
