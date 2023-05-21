import cv2 
import os
import ScrewPredictor as sp

image_folder_path = 'ScrewTrain/original_images'
image_names = os.listdir(image_folder_path)
screw_predictor = sp.ScrewPredictor(model_path='model/best.pt')
index = 1019
for image_name in image_names:
    image_path = os.path.join(image_folder_path, f'screw_NG_{index}.png')
    image = cv2.imread(image_path)

    screw_detection_result = screw_predictor.predict(image, conf=0.5)
    predicted_image = screw_predictor.getPredictedImage()

    # print(screw_detection_result)
    cv2.imshow('Predicted Image', predicted_image)
    cv2.waitKey(0)
    index = index + 1
    