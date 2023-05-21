import ultralytics
import os
import cv2

class ScrewPredictor():
  
  def __init__(self, **kwargs):
    model_path = kwargs['model_path'] if 'model_path' in kwargs else self.__getDefaultModelPath() 
    self.__model = ultralytics.YOLO(model=model_path, task='detect')
    self.input_image = None 
    self.predicted_result_from_model = None
    self.predicted_image = None

  def predict(self, input_image, **kwargs):
    '''
      The method predict bounding box and apply particular annalysis for anomoly screw detection.

      Args:
        image (numpy): image shape (original_width, original_height, channel).
        **kwargs: The keyword arguments are used for modification parameters

      Attributes:
        conf: confidence score, take bounding box having got confidence score greater than conf.

      Returns:
        screw_detection_result (dict): {'label': G/NG, 'prob': 0-1 (thread_probs)}

      PROGRESS:
        denoise image -> predict bounding box -> filter, analyse bounding box -> estimate anomaly and return
    '''

    confi_score = kwargs['conf'] if 'conf' in kwargs else 0.5

    self.input_image = input_image
    denoised_image = self.__preprocessImage(self.input_image)
    self.predicted_result_from_model = self.__model.predict(denoised_image[:,:,::-1], imgsz=128, conf=confi_score)[0]
    
    predicted_boxes = self.predicted_result_from_model.boxes
    print(predicted_boxes)
    screw_detection_result = {'label': None, 'prob': None}
    screw_detection_result['label'] = 'NG' if 1. in predicted_boxes.cls else 'G' 
    screw_detection_result['prob'] = 1.0 if screw_detection_result['label'] == 'G' else predicted_boxes.conf[predicted_boxes.cls[:] == 1.].cpu().numpy()[0]
    return screw_detection_result

  def getPredictedImage(self):
    '''
      The method draw predicted bounding box into input image for visualization.
      Must call after predict method have done.

      Args:
        None

      Attributes:
        None

      Returns:
        predicted_image (numpy): image shape (width, height, channel)
    '''

    predicted_boxes = self.predicted_result_from_model.boxes
    predicted_image = self.input_image
    original_image_shape = predicted_image.shape

    for index in range(0, predicted_boxes.xyxyn.shape[0]):
      width = original_image_shape[0]
      heigth = original_image_shape[1]
      start_point = predicted_boxes.xyxyn[index].cpu().numpy()[0:2]
      end_point = predicted_boxes.xyxyn[index].cpu().numpy()[2:]
      start_point[0] = start_point[0] * heigth
      end_point[0] = end_point[0] * heigth
      start_point[1] = start_point[1] * width
      end_point[1] = end_point[1] * width

      predicted_image = cv2.rectangle(predicted_image, start_point.astype(int).tolist(), end_point.astype(int).tolist(), (0, 0, 255), 1)
      predicted_image = cv2.putText(predicted_image, f'{self.predicted_result_from_model.names[int(predicted_boxes.cls[index])]}', start_point.astype(int).tolist(), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
  
    return predicted_image

  def __getDefaultModelPath(self):
    self.___default_model_folder = '/content/drive/MyDrive/Demo/TrainingReportYolov8/Screw_Yolov8s/runs/detect/train7/weights' 
    return os.path.join(self.___default_model_folder, 'best.pt')

  def __preprocessImage(self, image):
    '''
      The method resize image into shape (original_width, original_height, channel) = (width, 128, 3)
      and then apply filtering using Median and Bileteral (low pass) filter

      Args:
        image (numpy): image shape (original_width, original_height, channel).

      Attributes:
        None

      Returns:
        denoised_image (numpy): image shape (width, 128, 3) - Note: 128/width = original_height/original_width.

      PROGRESS:
        denoise image -> resize image and return.
    '''

    raw_image = image
    denoised_image = cv2.fastNlMeansDenoisingColored(raw_image, None, 10, 10, 7, 21)
    denoised_image = cv2.bilateralFilter(denoised_image, 9, 25, 25)

    original_shape = denoised_image.shape
    ratio = float(original_shape[0]) / float(original_shape[1])
    resized_hegiht = 128
    resized_width = int(resized_hegiht / ratio)
    dim = (resized_width, resized_hegiht)

    resized_image = cv2.resize(denoised_image, dim, interpolation=cv2.INTER_CUBIC)  
    return resized_image 