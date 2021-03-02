import pyrealsense2 as rs

class DeviceManager:
    def __init__(self, playback_file=None, enable_ir_emitter=True, width=1280, height=720, fps=30):
        """
        Class to manage the Intel RealSense devices

        Parameters:
        -----------
        """

        # intial variables
        self.enable_ir_emitter = enable_ir_emitter
        self.fps = fps 
        self.device_started = 0
        self.playback_file = playback_file

        # filters 
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2)
        colorizer = rs.colorizer()
        self.filters = [rs.decimation_filter(),
                   rs.disparity_transform(True),
                   rs.spatial_filter(),
                   rs.temporal_filter(), 
                   rs.disparity_transform(False),
                   rs.hole_filling_filter()]

        # configure realsense camera 
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # read from playback file if provided
        if self.playback_file:
            rs.config.enable_device_from_file(self.config, self.playback_file, repeat_playback=True) 

        # variables
        self.pipeline = None
        self.depth_frame = None
        self.color_frame = None
        self.depth_frame_filtered = None
        self.depth_sensor = None
        self.pixel_to_meters_scaling = 0
        self.depth_intrinsics = None
        self.w = 0
        self.h = 0

    def startDevice(self): 
        """
        Enable an Intel RealSense Device
        """

        # start the realsense pipeline
        self.pipeline = rs.pipeline()
        self.cfg = self.pipeline.start(self.config)

        # get intrinsics
        profile = self.cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
        self.depth_intrinsics = profile.as_video_stream_profile().get_intrinsics()
        self.depth_sensor = self.cfg.get_device().first_depth_sensor()
        self.pixel_to_meters_scaling = self.depth_sensor.get_depth_scale()

        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        self.device_started = 1

    def stopDevice(self): 
        """
        Turn off device
        """
        self.toggleEmitter(0)
        self.pipeline.stop()

    def getFrames(self): 
        """
        Obtain depth and rgb aligned frames 
        """

        # create necessary realsense objects
        align = rs.align(rs.stream.color)

        # This object configures the streaming camera and owns it's handle
        unaligned_frames = self.pipeline.wait_for_frames()
        
        # obtain aligned depth and color images   
        frames = align.process(unaligned_frames)
        self.depth_frame = frames.get_depth_frame()         
        self.color_frame = frames.get_color_frame()
        self.depth_frame_filtered = self.depth_frame
            
        # skip if not any are missing 
        if not self.depth_frame or not self.color_frame:
            return 0

        # get color intrinsics
        self.color_intrinsics = self.color_frame.profile.as_video_stream_profile().intrinsics

        # post process filtering for depth image 
        for f in self.filters:
            self.depth_frame_filtered = f.process(self.depth_frame_filtered)
        self.depth_frame_filtered = self.depth_frame_filtered.as_depth_frame() 
        self.depth_intrinsics = self.depth_frame_filtered.profile.as_video_stream_profile().intrinsics
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        return 1

    def toggleEmitter(self, emitter_toggle): 
        """
        Toggle laser emitter on and off
        """

        assert(self.device_started)
        self.depth_sensor.set_option(rs.option.emitter_enabled, emitter_toggle)