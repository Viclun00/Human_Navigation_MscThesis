import pyrealsense2 as rs
import cv2
import numpy as np
import math
from ultralytics import YOLO
from PIL import Image
import os


def get_centroid(mask_image):
  

    # Calculate moments
    moments = cv2.moments(mask_image)

    # Check for zero moment (avoid division by zero)
    if moments['m00'] == 0:
        centroid_x = centroid_y = 0.0
    else:
        # Extract centroid coordinates
        centroid_x = moments['m10'] / moments['m00']
        centroid_y = moments['m01'] / moments['m00']

    return (int(centroid_x), int(centroid_y))

def coord_toImg( x,y):
        x = round(x,1)
        y = round(y,1)
        x_translation = 5
        y_translation = 12
        x_scale = 2
        y_scale = 2
         

        x_transformed = int(x * x_scale + x_translation)
        y_transformed = int(y * y_scale + y_translation)

        return x_transformed,y_transformed
    
def add_toMap(x,y,size,clasif, binary_array = None):
	if binary_array is None:
		binary_array = (image_data > 50).astype(int)
	
	x_r, y_r = coord_toImg(x,y)
	if size != 0:
		for x in range(x_r-size, x_r+size):
			for y in range(y_r-size, y_r+size):
				if 0 <= x < binary_array.shape[0] and 0 <= y < binary_array.shape[1]:
					binary_array[x, y] = clasif
	else:
		x = x_r
		y = y_r
		if 0 <= x < binary_array.shape[0] and 0 <= y < binary_array.shape[1]:
					binary_array[x, y] = clasif
	
            
        
	return binary_array

def savePlot(binary_array):
	# Calculate the dimensions of the layout
	num_rows = 1
	num_cols = 1

	# Calculate the dimensions of each binary array
	array_height, array_width = binary_array.shape
	 
	# Calculate the dimensions of the combined image
	combined_height = array_height * num_rows
	combined_width = array_width * num_cols

	# Create an empty combined array
	combined_array = np.zeros((combined_height, combined_width), dtype=np.uint8)

	colors = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
	combined_image = Image.fromarray(combined_array).convert('RGB')

	height, width = combined_array.shape

	for y in range(height):
		for x in range(width):
			value = binary_array[y, x]
			combined_image.putpixel((x, y), colors[value])


	opencv_image = np.array(combined_image)
	opencv_image = opencv_image[:, :, ::-1].copy()
	cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Image Window', 600, 600)
	cv2.imshow('Image Window', opencv_image)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows() 


	
	image.save("current_map.png")
	

		
	
       

mir_pos = [0,0]

image = Image.open('Small_map.png')
image = image.convert('L')
image_data = np.array(image)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
print('[INFO] Starting streaming…')
print('[INFO] Camera ready.')

model = YOLO('yolov8n-seg.pt') #Load model

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()



while True:
	frames = pipeline.wait_for_frames() #Get images from camera
	color_frame = frames.get_color_frame()
	depth = frames.get_depth_frame()
	color_image = np.asanyarray(color_frame.get_data())

	depth_intrin = depth.profile.as_video_stream_profile().intrinsics
	results = model.predict(color_image, stream=True, conf=0.7, classes = [0])
	
	for r in results:
		masks = r.masks
		boxes = r.boxes
		center = []
		binary_array = (image_data > 50).astype(int)

		if masks and boxes:
			for mask in masks: #Calculate centroid and distance
				msk = mask.data[0].numpy()
				center = get_centroid(msk)
				zDepth = depth.get_distance(int(center[0]),int(center[1]))
				depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, center, zDepth)
 
				# bounding box for text
				binary_array = add_toMap(-depth_point[2]+mir_pos[0], depth_point[0]+mir_pos[1], 0, 4,binary_array)
				font = cv2.FONT_HERSHEY_SIMPLEX
				fontScale = 0.8
				color = (255, 100, 0)
				thickness = 2
				cv2.circle(color_image, (center[0], center[1]), 5, (0, 255, 0), 5)

				cv2.putText(color_image, f"Dist:{zDepth:.1f} x = {depth_point[0]:.1f} y = {depth_point[2]:.1f}", center, font, fontScale, color, thickness)

		# Display the resulting frame
	
	cv2.imshow('Webcam', color_image)
	savePlot(binary_array)
	


	if cv2.waitKey(1) == ord('q'):
			break

cv2.destroyAllWindows()