import airsim
import numpy as np
import cv2
import os
import time
import tempfile
import pandas as pd
# Connect to the AirSim simulator

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
def main():
	client = airsim.CarClient()
	client.confirmConnection()


	params = {'image': ['x_val',
		'y_val',
		'z_val',
	'Speed',
	'Gear',
	'Throttle',
	'Brake',
	'Steering']}




	i = 1

	try:
		while True:
    		# Get the car state
			car_state = client.getCarState()
			car_controls = airsim.CarControls()
			# Get the car's position and parameters
			position = car_state.kinematics_estimated.position
			speed = car_state.speed
			gear = car_state.gear
			throttle = car_controls.throttle
			brake = car_controls.brake
			steering = car_controls.steering
			responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False), airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
			response = responses[0]
			response1 = responses[1]

			# get numpy array
			img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

			# reshape array to 4 channel image array H X W X 4
			img_rgb = img1d.reshape(response.height, response.width, 3)


			# write to png 
			airsim.write_png(os.path.join(tmp_dir, 'segment_'+str(i)+'.png'), img_rgb) 

			img1d = np.fromstring(response1.image_data_uint8, dtype=np.uint8) 

			# reshape array to 4 channel image array H X W X 4
			img_rgb = img1d.reshape(response1.height, response1.width, 3)


			# write to png 
			airsim.write_png(os.path.join(tmp_dir, 'scene_'+str(i)+'.png'), img_rgb) 

			# image = np.frombuffer(image_data, dtype=np.uint8).reshape(response[0].height, response[0].width, -1)
			# image_path = "car_image.png"

			# Save the image and print the car parameters
			# cv2.imwrite(image_path, image)
			print("Car Parameters:")
			print("Position:", position)
			print("Speed:", speed)
			print("Gear:", gear)
			print("Throttle:", throttle)
			print("Brake:", brake)
			print("Steering:", steering)

			liste = [position.x_val, position.y_val, position.z_val, speed, gear, throttle, brake, steering]
			params.update({i: liste})
			i += 1
			time.sleep(0.2)


	except KeyboardInterrupt:
		pass
	print(params)
	df = pd.DataFrame.from_dict(params)
	df.T.to_csv(os.path.join(tmp_dir,'parameters.csv'), sep=';')

#main()