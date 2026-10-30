
from ximea import xiapi
import time

syncGPIO = True

xiCam0 = xiapi.Camera()
xiCam0.open_device_by_SN('39650151')
# xiCam1.open_device_by_SN('39650051')

if syncGPIO:
    # Set triggers to synchronize with video-projectors framerate
    print('Sync settings : XI_TRG_EDGE_RISING')
    xiCam0.set_gpi_selector('XI_GPI_PORT1')                           # Select input 1 as trigger
    xiCam0.set_trigger_source('XI_TRG_EDGE_RISING')  # Select trigger source
    xiCam0.set_gpi_mode('XI_GPI_TRIGGER')
    # xiCam0.set_gpo_selector('XI_GPO_PORT1')                         # Set digital output 1 mode
    # xiCam0.set_gpo_mode('XI_GPO_EXPOSURE_ACTIVE')

    # xiCam1.set_gpi_selector('XI_GPI_PORT1')                           # Select input 1 as trigger
    # xiCam1.set_trigger_source('XI_TRG_EDGE_RISING')  # Select trigger source
    # xiCam1.set_gpi_mode('XI_GPI_TRIGGER')
    # # xiCam1.set_gpo_selector('XI_GPO_PORT1')                         # Set digital output 1 mode
    # # xiCam1.set_gpo_mode('XI_GPO_EXPOSURE_ACTIVE')

else:
    # Set framerate (use camera's internal clock)
    print('Sync settings : fixed framerate')
    xiCam0.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
    xiCam0.set_framerate(60)
    # xiCam1.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
    # xiCam1.set_framerate(60)

# Set camera parameters
xiCam0.set_exposure(16500)
xiCam0.set_gain(1)
xiCam0.set_param('recent_frame', 0)
xiCam0.enable_auto_wb()
# xiCam0.set_acq_transport_buffer_commit(32)
# xiCam0.set_acq_buffer_size(xiCam.get_acq_buffer_size_maximum())
xiCam0.set_imgdataformat('XI_RGB24')

# Prepare ROI crop ranges
# Create instance of Image to store image data and metadata
xiImg = xiapi.Image()

# Start data acquisition
xiCam0.start_acquisition()

time.sleep(1)

for i in range(100):
    print('Waiting for image %d...' % i)

    xiCam0.get_image(xiImg)
    print('Received')

xiCam0.stop_acquisition()
xiCam0.close_device()
