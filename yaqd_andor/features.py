from collections import namedtuple

Spec = namedtuple("Feature", ["sdk_name", "type", "availability"])

# adapted from sdk3 manual
# note some table entries were omitted (focused on neo, simcam features)
# a = apogee, n = neo, s = simcam, z = zyla
specs = dict(
    acquisition_start = Spec("AcquisitionStart", "command", "nsz"),
    acquisition_stop = Spec("AcquisitionStop", "command", "nsz"),
    aoi_binning = Spec("AOIBinning", "enumerated", "nz"),
    aoi_hbin = Spec("AOIHBin", "integer", "asz"),
    aoi_height = Spec("AOIHeight", "integer", "ansz"),
    aoi_layout = Spec("AOILayout", "enumerated", "az"),
    aoi_left = Spec("AOILeft", "integer", "ansz"),
    aoi_stride = Spec("AOIStride", "integer", "nz"),
    aoi_top = Spec("AOITop", "integer", "ansz"),
    aoi_vbin = Spec("AOIVBin", "integer", "asz"),
    aoi_width = Spec("AOIWidth", "integer", "ansz"),
    baseline = Spec("Baseline", "integer", "nz"),
    bit_depth = Spec("BitDepth", "enumerated", "anz"),
    buffer_overflow_event = Spec("BufferOverflowEvent", "integer", "nz"),
    bytes_per_pixel = Spec("BytesPerPixel", "float", "nz"),
    camera_acquiring = Spec("CameraAcquiring", "boolean", "nsz"),
    camera_dump = Spec("CameraDump", "command", "nz"),
    camera_family = Spec("CameraFamily", "string", "a"),
    camera_memory = Spec("CameraMemory", "integer", "a"),
    camera_model = Spec("CameraModel", "string", "nsz"),
    camera_name = Spec("CameraName", "string", "anz"),
    cycle_mode = Spec("CycleMode", "enumerated", "nsz"),
    electronic_shuttering_mode = Spec("ElectronicShutteringMode", "enumerated", "nsz"),
    event_enable = Spec("EventEnable", "boolean", "nz"),
    events_missed_event = Spec("EventsMissedEvent", "integer", "nz"),
    event_selector = Spec("EventSelector", "enumerated", "nz"),
    exposure_time = Spec("ExposureTime", "float", "nsz"),
    exposure_end_event = Spec("ExposureEndEvent", "integer", "nz"),
    exposure_start_event = Spec("ExposureStartEvent", "integer", "nz"),
    external_trigger_delay = Spec("ExposureTriggerDelay", "float", "z"),
    fan_speed = Spec("FanSpeed", "enumerated", "ansz"),
    # fast_aoi_frame_rate_enable = Spec("FastAOIFrameRateEnable", "boolean", "nz"),
    firmware_version = Spec("FirmwareVersion", "string", "anz"),
    frame_count = Spec("FrameCount", "integer", "ansz"),
    frame_rate = Spec("FrameInterval", "float", "ansz"),
    full_aoi_control = Spec("FullAOIControl", "boolean", "nz"),
    image_size_bytes = Spec("ImageSizeBytes", "integer", "nsz"),
    interface_type = Spec("InterfaceType", "string", "anz"),
    max_interface_transfer_rate = Spec("MaxInterfaceTransferRate", "float", "nz"),
    metadata_enable = Spec("MetadataEnable", "boolean", "nz"),
    metadata_frame = Spec("MetadataFrame", "boolean", "nz"),
    metadata_timestamp = Spec("MetadataTimestamp", "boolean", "nz"),
    overlap = Spec("Overlap", "boolean", "anz"),
    pixel_correction = Spec("PixelCorrection", "enumerated", "s"),
    pixel_encoding = Spec("PixelEncoding", "enumerated", "nsz"),
    pixel_height = Spec("PixelHeight", "enumerated", "nsz"),
    pixel_readout_rate = Spec("PixelReadoutRate", "enumerated", "ansz"),
    pixel_width = Spec("PixelWidth", "float", "ansz"),
    readout_time = Spec("ReadoutTime", "float", "nz"),
    sensor_cooling = Spec("SensorCooling", "boolean", "ansz"),
    sensor_height = Spec("SensorHeight", "integer", "ansz"),
    sensor_readout_mode = Spec("SensorReadoutMode", "enumerated", "z"),
    sensor_temperature = Spec("SensorTemperature", "float", "ansz"),
    sensor_width = Spec("SensorWidth", "integer", "ansz"),
    serial_number = Spec("SerialNumber", "string", "snz"),
    simple_preamp_gain_control = Spec("SimplePreampGainControl", "enumerated", "nz"),
    software_trigger = Spec("SoftwareTrigger", "command", "nz"),
    spurious_noise_filter = Spec("SpuriousNoiseFilter", "boolean", "nz"),
    static_blemish_correction = Spec("StaticBlemishCorrection", "boolean", "nz"),
    temperature_control = Spec("TemperatureControl", "enumerated", "n"),
    temperature_status = Spec("TemperatureStatus", "enumerated", "anz"),
    timestamp_clock = Spec("TimestampClock", "integer", "nz"),
    timestamp_clock_frequency = Spec("TimestampClockFrequency", "integer", "nz"),
    timestamp_clock_reset = Spec("TimestampClockReset", "command", "nz"),
    trigger_mode = Spec("TriggerMode", "enumerated", "ansz"),
    vertically_center_aoi = Spec("VerticallyCentreAOI", "boolean", "nz")
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
        self.method = r"{}" + f"_{type}"
        # feature implemented?
        self.is_implemented = self.sdk.is_implemented(self.hndl, self.sdk_name)
        if self.is_implemented:
                self.get = self._get
        else:
            raise NotImplementedError(f"feature {self.sdk_name} is not implemented")
        # is read only?
        self.is_readonly = self.sdk.is_readonly(self.hndl, self.sdk_name)
        if not self.is_readonly:
            self.set = self._set
        self._get_call = f"get_{self.type}"

    def _get(self):
        call = self._get_call
        if self.sdk.is_readable(self.hndl, self.sdk_name):
            return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)
        else:
            raise TypeError(f"call {call} is not readable")

    def _set(self, value):
        call = self._set_call
        if self.sdk.is_writable(self.hndl, self.sdk_name):
            return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name, value)
        else:
            raise ValueError(f"call {call} is not currently writable")    


class SDKString(Feature):
    def __init__(self, *args):
        super().__init__(*args)


class SDKFloat(Feature):
    def __init__(self, *args):
        super().__init__(*args)

    def max(self) -> float:
        call = self.get_call + "_max"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)

    def min(self) -> float:
        call = self.get_call + "_min"
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
        del self._set
        del self._get

    def __call__(self):
        return self.sdk.command(self.hndl, self.sdk_name)


class SDKEnum(Feature):
    def __init__(self, *args):
        super().__init__(*args)
        self._get_call = self._get_call + "_string"
        self._set_call = self._set_call + "_string"

    def options(self):
        """query available feature string options
        """
        return self.sdk.get_enumerated_string_options(self.hndl, self.sdk_name)

