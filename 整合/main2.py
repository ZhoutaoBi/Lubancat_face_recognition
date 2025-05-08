import pandas as pd
import numpy as np
import cv2 as cv

import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from copy import copy
import time
import datetime

import json
import time
import paho.mqtt.client as mqtt
from MqttSign import AuthIfo
from ringbuff import RingBuffer
import threading


flag_bi=0
number_bi = 0
time_bi = 0
flag_yolo = 1
flag_people = 0

# set the device info, include product key, device name, and device secret
productKey = "guth3vOknf0"
deviceName = "lubancat_face"
deviceSecret = "9f19c035edb6eb5b5293f3f4bd9b6838"

# set timestamp, clientid, subscribe topic and publish topic
timeStamp = str((int(round(time.time() * 1000))))
clientId = "lubancat"
pubTopic = "/sys/" + productKey + "/" + deviceName + "/thing/event/property/post"
subTopic = "/sys/" + productKey + "/" + deviceName + "/thing/event/property/post_reply"

# set host, port
host = productKey + ".iot-as-mqtt.cn-shanghai.aliyuncs.com"
# instanceId = "***"
# host = instanceId + ".mqtt.iothub.aliyuncs.com"
port = 1883
keepAlive = 300

# calculate the login auth info, and set it into the connection options
m = AuthIfo()
m.calculate_sign_time(productKey, deviceName, deviceSecret, clientId, timeStamp)
client = mqtt.Client(m.mqttClientId)
client.username_pw_set(username=m.mqttUsername, password=m.mqttPassword)

def create_alink_json(name, value):
    """
    创建符合阿里云 ALINK 协议的 JSON 消息

    :param name: 传感器名称
    :param value: 传感器的测量值
    :return: 生成的 JSON 字符串，如果创建失败返回 None
    """
    # 检查输入参数是否有效
    if not name or not value:
        return None

    # 构建 JSON 结构
    alink_json = {
        "id": "123",
        "version": "1.0",
        "sys": {
            "ack": 0
        },
        "params": {
            name: {
                "value": value
            }
        },
        "method": "thing.event.property.post"
    }

    try:
        # 将字典转换为 JSON 字符串
        result = json.dumps(alink_json)
        return result
    except Exception as e:
        print(f"创建 JSON 时出错: {e}")
        return None

def face_detect_thread():
    """
    传感器线程函数，
    """
    face_detect_data = {
        "id": "123",
        "version": "1.0",
        "sys": {"ack": 0},
        "params": {"face": {"value": face_detect}},
        "method": "thing.event.property.post"
    }

    while True:
        result = ring_buffer.write(face_detect_data)
        if result == 0:
            # print("[face_detect Thread] 成功将数据写入环形缓冲区")
            pass
        else:
            print("[face_detect Thread] 环形缓冲区已满，无法写入数据")
            pass
        # 模拟传感器数据更新间隔
        threading.Event().wait(2)

def on_connect(client, userdata, flags, rc):
    """
    MQTT 连接回调函数，处理连接成功或失败的情况

    :param client: MQTT 客户端对象
    :param userdata: 用户数据
    :param flags: 连接标志
    :param rc: 连接结果码
    """
    if rc == 0:
        print("Connect aliyun IoT Cloud Sucess")
    else:
        print("Connect failed...  error code is:" + str(rc))

def on_message(client, userdata, msg):
    """
    MQTT 消息接收回调函数，处理接收到的消息

    :param client: MQTT 客户端对象
    :param userdata: 用户数据
    :param msg: 接收到的消息对象，包含主题和负载
    """
    topic = msg.topic
    payload = msg.payload.decode()
    print("receive message ---------- topic is : " + topic)
    print("receive message ---------- payload is : " + payload)
    print("\n")

def connect_mqtt():
    """
    连接到 MQTT 服务器

    :return: MQTT 客户端对象
    """
    client.connect(host, port, keepAlive)
    return client

def publish_message():
    """
    从环形缓冲区读取数据并发布到 MQTT 服务器的函数
    """
    #while True:
    data = ring_buffer.read()
    if data is not None:
            # print("[MQTT Send Thread] 从环形缓冲区读取数据:", data)
        client.publish(pubTopic, json.dumps(data))
        
    else:
        print("[MQTT Send Thread] 环形缓冲区为空，无数据可读取")
        # 模拟 MQTT 发送间隔
    #threading.Event().wait(1)

def subscribe_topic():
    """
    订阅 MQTT 主题
    """
    # subscribe to subTopic("/a1LhUsK****/python***/user/get") and request messages to be delivered
    client.subscribe(subTopic)
    print("subscribe topic: " + subTopic)


RKNN_MODEL = './model/yolov5s.rknn'

QUANTIZE_ON = True

OBJ_THRESH = 0.45
NMS_THRESH = 0.45
IMG_SIZE = 640
BOX = (450, 150, 1100, 550)
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y



def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!
    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.
    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
    
    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)


    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        if CLASSES[cl] == "person":
            #print('class: {}, score: {}'.format(CLASSES[cl], score))
            #print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            top = int(top)
            left = int(left)
            right = int(right)
            bottom = int(bottom)



def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def get_real_box(src_shape, box, dw, dh, ratio):
    bbox = copy(box)
    # unletter_box result
    bbox[:,0] -= dw
    bbox[:,0] /= ratio
    bbox[:,0] = np.clip(bbox[:,0], 0, src_shape[1])
 
    bbox[:,1] -= dh
    bbox[:,1] /= ratio
    bbox[:,1] = np.clip(bbox[:,1], 0, src_shape[0])
 
    bbox[:,2] -= dw
    bbox[:,2] /= ratio
    bbox[:,2] = np.clip(bbox[:,2], 0, src_shape[1])
 
    bbox[:,3] -= dh
    bbox[:,3] /= ratio
    bbox[:,3] = np.clip(bbox[:,3], 0, src_shape[0])
    return bbox


if __name__ == '__main__':

    ring_buffer = RingBuffer(10)
    # Set the on_connect callback function for the MQTT client
    client.on_connect = on_connect
    # Set the on_message callback function for the MQTT client
    client.on_message = on_message
    client = connect_mqtt()
    # Start the MQTT client loop in a non-blocking manner
    client.loop_start()

    #subscribe_topic()
    

    print('NO.1 lbph loading...........')
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]
    faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')
    lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)
    lbph.read('Classifiers/TrainedLBPH.yml')
    print('NO.1 lbph loaded successfully')

    print('NO.2 rknn loading...........')
    rknn_lite = RKNNLite()
    # Load RKNN model
    print('--> Loading model')
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    print('NO.2 rknn loaded successfully')

    print('NO.3 camera loading...........')

    camera = cv.VideoCapture(9)
    print('NO.3 camera loaded successfully')

    face_detect = 'start record'
    face_detect_thread_obj = threading.Thread(target=face_detect_thread)
    face_detect_thread_obj.daemon = True
    face_detect_thread_obj.start()
    publish_message()

    data_to_x_send = "200"
    with open('shared_x_file.txt', 'w') as file:
        file.write(data_to_x_send)
    data_to_y_send = "100"
    with open('shared_y_file.txt', 'w') as file:
        file.write(data_to_y_send)



    while cv.waitKey(1) & 0xFF != ord('q'):

        

        _, img = camera.read()
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)
        print(faces)
        if str(faces) == '()':
            flag_yolo = 1
            print("yolo is open")
        else:
            flag_yolo = 0
            print("yolo is close") 
        for x, y, w, h in faces:
            faceRegion = grey[y:y + h, x:x + w]
            faceRegion = cv.resize(faceRegion, (220, 220))
            label, trust = lbph.predict(faceRegion)


    
        src_shape = img.shape[:2]
        img_yolo, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
        img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
        img_yolo2 = np.expand_dims(img_yolo, 0)

        # Inference
        if flag_yolo == 1:
    
            outputs = rknn_lite.inference(inputs=[img_yolo2], data_format=['nhwc'])

            # post process
            input0_data = outputs[0]
            input1_data = outputs[1]
            input2_data = outputs[2]

            input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
            input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
            input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

            input_data = list()
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

            boxes, classes, scores = yolov5_post_process(input_data)

            if boxes is not None:
                cv.putText(img, "have people", (320,60), cv.FONT_HERSHEY_COMPLEX, 1.5, (238, 238, 0),2)
                print('have people...')
                draw(img, boxes, scores, classes)
                flag_people = 1
                
            else:
                cv.putText(img, "no people", (340,60), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255),2)
                print('no people...')
                time_bi+=1
                if time_bi >10:
                    flag_bi = 0
                    number_bi = 0
                    time_bi = 0 
                with open('shared_x_file.txt', 'w') as file:
                    file.write(data_to_x_send)
                data_to_y_send = str(y)
                with open('shared_y_file.txt', 'w') as file:
                    file.write(data_to_y_send)

        try:
            if (100 - trust) > 35 :
                name = id_names[id_names['id'] == label]['name'].item()
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(img, "have people", (320,60), cv.FONT_HERSHEY_COMPLEX, 1.5, (238, 238, 0),2)
                cv.putText(img, name, (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                print(name,trust)
                print(x,'+',y)
                data_to_x_send = str(x)
                with open('shared_x_file.txt', 'w') as file:
                    file.write(data_to_x_send)
                data_to_y_send = str(y)
                with open('shared_y_file.txt', 'w') as file:
                    file.write(data_to_y_send)
                number_bi +=1
                #if  (x-210)>15:
                #    set_servo_angle((x-210))
            else:
                if trust!= 100:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(img, "have people", (320,60), cv.FONT_HERSHEY_COMPLEX, 1.5, (238, 238, 0),2)
                    cv.putText(img, 'others', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                    number_bi -=1
                    print('others',trust)
                    print(x,'+',y)
            trust = 100
        except:
            pass

        if number_bi >4:
            flag_bi = 1
            number_bi = 4
            face_detect = 'is ZhoutaoBi'
            face_detect_thread_obj = threading.Thread(target=face_detect_thread)
            face_detect_thread_obj.daemon = True
            face_detect_thread_obj.start()
            publish_message()

        if number_bi <-4:
            flag_bi = -1
            number_bi = -4
            face_detect = 'WARNING'
            face_detect_thread_obj = threading.Thread(target=face_detect_thread)
            face_detect_thread_obj.daemon = True
            face_detect_thread_obj.start()
            publish_message()

        if flag_people == 1:
            face_detect = 'have people'
            face_detect_thread_obj = threading.Thread(target=face_detect_thread)
            face_detect_thread_obj.daemon = True
            face_detect_thread_obj.start()
            publish_message()
            flag_people = 0

        if flag_bi == 1:
            cv.putText(img, "GOOD", (320,400), cv.FONT_HERSHEY_COMPLEX, 1.5, (238, 238, 0),2)
            print('The correct ID is successfully verified')
            #获取当前日期和时间,类型：datetime
            now_time = datetime.datetime.now()
            print(now_time)
        if flag_bi == -1:
            cv.putText(img, "WARNING", (320,400), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255),2)
            print('Personnel have been found to have stayed too long')
            #获取当前日期和时间,类型：datetime
            now_time = datetime.datetime.now()
            print(now_time)
        cv.imshow('Recognize', img)

    camera.release()
    rknn_lite.release()
    cv.destroyAllWindows()
