"""

Inference engine detector sample

python practive2_template.py -c ../models/mobilenet-ssd.xml -w ../models/mobilenet-ssd.bin -i ../images/dog.jpg -d CPU -l "C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"
 
"""
import sys
import cv2
import argparse
import logging as log
sys.path.append('../src')
from ie_detector import InferenceEngineDetector

def build_argparse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', help = 'Path to .xml model file', 
                        type = str)
    parser.add_argument('-w', '--weights', help = 'Path to .bin model file', 
                        type = str)
    parser.add_argument('-l', '--cpu_extension', help = 'Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.',    
                        default = '', type = str)
    parser.add_argument('-d', '--device', help = 'Execution device', 
                        type=str, default = 'CPU')
    parser.add_argument('-i', '--input', help = 'Input argument', 
                        type = str)
    return parser

def main():
    log.basicConfig(format = '[%(levelname)s] %(message)s', level = log.INFO, 
                               stream = sys.stdout)
    log.info('Hello object detection!')
    args = build_argparse().parse_args()
    
    image_src = cv2.imread(args.input)
    log.info('Loaded image with size {}'.format(image_src.shape))
    
    detector = InferenceEngineDetector(args.weights, args.config, args.device, args.cpu_extension)
    log.info('Loaded dl net')
    
    detect_result = detector.detect(image_src)
    log.info('Objects are detected')
    
    detect_image = detector.draw_detection(detect_result, image_src)
    log.info('Objects are drown')
    
    cv2.imshow('Detections', detect_image)
    cv2.waitKey(0) # waits until a key is pressed

    cv2.destroyAllWindows() 
    
    return 

if __name__ == '__main__':
    sys.exit(main()) 