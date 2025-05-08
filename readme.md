## 2025.5.1
update the compelete codes.

Abstract: This research focuses on the design of face recognition and tracking system based on RK3566, aiming to build an efficient and practical intelligent monitoring system, and build a three-level architecture of "device, edge and cloud". Research on the integration of multi-disciplinary technologies such as deep learning and Internet of Things communication. The system hardware platform is equipped with a quad-core Cortex-A55 processor (main frequency 1.8GHz) and a NPU module with 0.8TOPS computing power to achieve dynamic tracking of X-axis -60° to 60° and Y-axis -30° to 40° of the 2-DOF steering gear head, and a high-definition camera module with integrated USB interface. At the algorithm level, this paper implemented NPU directional optimization strategy in YOLOv5s target detection model, completed model quantization and operator fusion by RKNN-Toolkit2, and realized detection frame rate increased to 11fps (360x360 resolution). In this paper, LBP texture histogram (256-dimensional) and HSV color space features are integrated to build a composite feature descriptor, and the accuracy of face recognition in well-lit scenes is 98%. In terms of communication protocol stack, this paper adopts lightweight MQTT Internet of Things protocol to establish end-to-end data transmission channel, and the average packet loss rate is less than 1%.
This research provides a reliable solution for the edge intelligent monitoring system, and its technical route can be extended to the field of behavior analysis, anomaly detection, etc., which can provide certain theoretical and technical support for the further development of intelligent monitoring system.





## 2025.1.5
change the yunet.onnx to new1_yunet.onnx
because the opencv need static parameter but yunet.onnx is dynamic parameter
./models/build_new_yunet.py 
./models/check_new_yunet.py 
