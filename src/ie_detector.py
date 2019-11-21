"""

Inference engine detector
 
"""
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class InferenceEngineDetector:
    def __init__(self, weightsPath = None, configPath = None,
                 device = 'CPU', extension = None):
        self.weights = weightsPath
        self.config = configPath
        self.ie = IECore()
        self.net = IENetwork(model = configPath, weights = weightsPath)
        if extension:
            self.ie.add_extension(extension, 'CPU')
        self.exec_net = self.ie.load_network(network = self.net, 
                                             device_name = device)
        return

    def draw_detection(self, detections, img):
        (h, w) = img.shape[:2]
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
        return img

    def _prepare_image(self, image, h, w):
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        return image
        
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        blob = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: blob})
        detection = output[out_blob]
        return detection