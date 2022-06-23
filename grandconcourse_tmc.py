#Grand Concourse Turn Count
# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh,increment_path
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from shapely.geometry import Polygon, LineString
import csv
import numpy as np


all_new_turn2_dict ={}
all_new_turn2_count = 0

all_new_turn3_dict ={}
all_new_turn3_count = 0

all_new_turn5_dict ={}
all_new_turn5_count = 0

all_new_turn12_dict ={}
all_new_turn12_count = 0

all_new_turn8_dict ={}
all_new_turn8_count = 0

all_new_turn11_dict ={}
all_new_turn11_count = 0


all_new_turn9_dict ={}
all_new_turn9_count = 0

all_new_turn6_dict ={}
all_new_turn6_count = 0




def convertToPoly(objDims):  # [x1, y1, x2, y2]
    box = objDims
    b1_x1, b1_y1, b1_x2, b1_y2 = box[0], box[1], box[2], box[3]
    poly = Polygon([(b1_x1, b1_y1), (b1_x2, b1_y1), (b1_x2, b1_y2), (b1_x1, b1_y2), (b1_x1, b1_y1)])
    return poly

rect1_ = [530, 300, 700, 450]
rect1_shrink = [530, 330, 700, 450]
rect2_ = [605, 200, 700, 260]
rect3_ = [220, 60, 340, 150]
rect4_ = [10, 230, 180, 450]
rect4_shrink = [10, 230, 150, 450]
rect5_ = [230, 350, 500, 450]
rect5_extended = [230, 300, 500, 450]
rect6_ = [10, 140, 60, 200]
rect7_ = [80, 100, 190, 160]
rect7_shrink = [80, 100, 190, 130]
rect8_ = [400, 100, 600, 200]




rect4_poly = convertToPoly(rect4_)
rect4_shrink_poly = convertToPoly(rect4_shrink)
rect2_poly = convertToPoly(rect2_)
rect5_poly = convertToPoly(rect5_)
rect5_extended_poly = convertToPoly(rect5_extended)
rect8_poly = convertToPoly(rect8_)
rect6_poly = convertToPoly(rect6_)
rect1_poly = convertToPoly(rect1_)
rect1_shrink_poly = convertToPoly(rect1_shrink)
poly1 = Polygon([(561, 376), (655, 306), (706, 361), (662, 452), (561, 376)])
rect7_poly = convertToPoly(rect7_)
rect3_poly = convertToPoly(rect3_)
rect7_shrink_poly = convertToPoly(rect7_shrink)
turn6_poly_enter = Polygon([(429, 148), (549, 142), (603, 194), (465, 190), (429, 148)])
turn6_poly_exit = Polygon([(177, 106), (240, 90), (334, 144), (224, 149), (177, 106)])



def turn2_count(x1, x2, y2, id):
    global all_new_turn2_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(rect4_poly).length
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn2_dict:
        all_new_turn2_dict[int(id)] = 'false'

    if int(id) in all_new_turn2_dict and all_new_turn2_dict[int(id)] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line2.intersection(rect2_poly).length
        length_full2 = the_line2.length
        if length_intersect2 / length_full2 > 0.4:
            all_new_turn2_dict[int(id)] = 'true'
            all_new_turn2_count += 1






def turn3_count(x1, x2, y2, id):
    global all_new_turn3_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(rect4_shrink_poly).length #enter
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn3_dict:
        all_new_turn3_dict[int(id)] = ['false', 'false'] #[didn't appear in rect5, didn't appear in rect2]

    if int(id) in all_new_turn3_dict and all_new_turn3_dict[int(id)][0] == 'false' and all_new_turn3_dict[int(id)][1] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line2.intersection(rect5_poly).length   #exit
        length_full2 = the_line2.length

        if length_intersect2 / length_full2 > 0.4:
            all_new_turn3_dict[int(id)][0] = 'true'
            all_new_turn3_count += 1

    if int(id) in all_new_turn3_dict and all_new_turn3_dict[int(id)][0] == 'true' and all_new_turn3_dict[int(id)][1] == 'false':
        the_line3 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line3.intersection(rect2_poly).length
        if length_intersect2>0:
            all_new_turn3_dict[int(id)][1] = 'true'
            all_new_turn3_count -= 1




def turn5_count(x1, x2, y2, id):
    global all_new_turn5_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(rect8_poly).length #enter
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn3_dict:
        all_new_turn5_dict[int(id)] = 'false'

    if int(id) in all_new_turn5_dict and all_new_turn5_dict[int(id)] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line.intersection(rect6_poly).length   #exit
        length_full2 = the_line2.length
        if length_intersect2 / length_full2 > 0.4:
            all_new_turn5_dict[int(id)] = 'true'
            all_new_turn5_count += 1



def turn8_count(x1, x2, y2, id):
    global all_new_turn8_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(rect7_poly).length #enter
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn8_dict:
        all_new_turn8_dict[int(id)] = 'false'

    if int(id) in all_new_turn8_dict and all_new_turn8_dict[int(id)] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line.intersection(rect5_extended_poly).length   #exit
        length_full2 = the_line2.length
        if length_intersect2 / length_full2 > 0.4:
            all_new_turn8_dict[int(id)] = 'true'
            all_new_turn8_count += 1


def turn11_count(x1, x2, y2, id):
    global all_new_turn11_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(rect1_poly).length #enter
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn11_dict:
        all_new_turn11_dict[int(id)] = 'false'

    if int(id) in all_new_turn11_dict and all_new_turn11_dict[int(id)] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line.intersection(rect3_poly).length   #exit
        length_full2 = the_line2.length
        if length_intersect2 / length_full2 > 0.4:
            all_new_turn11_dict[int(id)] = 'true'
            all_new_turn11_count += 1

def turn9_count(x1, x2, y2, id):
    global all_new_turn9_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(rect7_shrink_poly).length #enter
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn9_dict:
        all_new_turn9_dict[int(id)] = 'false'

    if int(id) in all_new_turn9_dict and all_new_turn9_dict[int(id)] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line2.intersection(rect6_poly).length   #exit
        length_full2 = the_line2.length
        if length_intersect2 / length_full2 > 0.4:
            all_new_turn9_dict[int(id)] = 'true'
            all_new_turn9_count += 1

def turn12_count(x1, x2, y2, id):
    global all_new_turn12_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(poly1).length #enter
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn12_dict:
        all_new_turn12_dict[int(id)] = 'false'

    if int(id) in all_new_turn12_dict and all_new_turn12_dict[int(id)] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line2.intersection(rect2_poly).length   #exit
        length_full2 = the_line2.length
        if length_intersect2 / length_full2 > 0.4:
            all_new_turn12_dict[int(id)] = 'true'
            all_new_turn12_count += 1

def turn6_count(x1, x2, y2, id):
    global all_new_turn6_count
    the_line = LineString([(x1, y2), (x2, y2)])
    length_intersect = the_line.intersection(turn6_poly_enter).length #enter
    length_full = the_line.length
    if length_intersect / length_full > 0.4 and int(id) not in all_new_turn6_dict:
        all_new_turn6_dict[int(id)] = 'false'

    if int(id) in all_new_turn6_dict and all_new_turn6_dict[int(id)] == 'false':
        the_line2 = LineString([(x1, y2), (x2, y2)])
        length_intersect2 = the_line2.intersection(turn6_poly_exit).length   #exit
        length_full2 = the_line2.length
        if length_intersect2 / length_full2 > 0.4:
            all_new_turn6_dict[int(id)] = 'true'
            all_new_turn6_count += 1






def detect(opt):
    #these are to draw on vid
    rect1 = [(530, 300), (700, 450)]
    rect2 = [(605, 200), (700, 260)]
    rect3 = [(220, 60), (340, 150)]
    rect4 = [(10, 230), (180, 450)]
    rect5 = [(230, 400), (500, 450)]
    rect6 = [(10, 140), (60, 200)]
    rect7 = [(80, 100), (190, 160)]
    rect8 = [(400, 100), (600, 200)]
    rect_mid = [(400, 200), (490, 250)]

    #global rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8
    global all_new_turn2_count, all_new_turn3_count, all_new_turn5_count, all_new_turn6_count, all_new_turn8_count, all_new_turn9_count, all_new_turn11_count, all_new_turn12_count
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, half = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.imgsz, opt.evaluate, opt.half
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(opt.yolo_weights, device=device, dnn=opt.dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    csv_path = str(Path(out)) + '/' + txt_file_name + '.csv'
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)


                #cv2.rectangle(im0, rect1[0], rect1[1], (0, 0, 0), 1) #1
                #cv2.rectangle(im0, rect2[0], rect2[1], (0, 0, 0), 1) #2
                #cv2.rectangle(im0, rect3[0], rect3[1], (0, 0, 0), 1) #3
                #cv2.rectangle(im0, rect4[0], rect4[1], (0, 0, 0), 1) #4
                #cv2.rectangle(im0, rect5[0], rect5[1], (0, 0, 255), 1) #5
                #cv2.rectangle(im0, rect6[0], rect6[1], (0, 0, 0), 1) #6
                #cv2.rectangle(im0, rect7[0], rect7[1], (0, 0, 0), 1) #7
                #cv2.rectangle(im0, rect8[0], rect8[1], (0, 0, 0), 1) #8

                pts0 = np.array([[561, 376], [655, 306], [706, 361], [662, 452]],np.int32)
                cv2.polylines(im0, [pts0], True, (0, 0, 255), 1)

                pts1 = np.array([[429, 148], [549, 142], [603, 194], [465, 190]], np.int32)
                cv2.polylines(im0, [pts1], True, (0, 0, 255), 1)

                pts2 = np.array([[177,106], [240,90], [334,144], [224,149], [177,106]], np.int32)
                cv2.polylines(im0, [pts2], True, (0, 0, 255), 1)


                pts3=np.array([[571,219], [645,217], [688,255], [663,275]],np.int32)
                cv2.polylines(im0, [pts3], True, (0, 0, 255), 1)

                pts4 = np.array([[8,257], [102,244], [137,468], [8,475]], np.int32)
                cv2.polylines(im0, [pts4], True, (0, 0, 255), 1)

                pts5 = np.array([[6,164], [69,153], [75,184], [10,190]], np.int32)
                cv2.polylines(im0, [pts5], True, (0, 0, 255), 1)

                pts6 = np.array([[82,95], [128,88], [183,131], [86,139]], np.int32)
                cv2.polylines(im0, [pts6], True, (0, 0, 255), 1)

                pts7 = np.array([[243,382], [481,343], [544,388], [316,463]], np.int32)
                cv2.polylines(im0, [pts7], True, (0, 0, 255), 1)

                #test rectangles
                #cv2.rectangle(im0, rect_mid[0], rect_mid[1], (0, 0, 230), 1)



            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        #if int(id) == 2: #use this to check for certain ids
                        annotator.box_label(bboxes, label, color=colors(c, True))




                        x1 = output[0]
                        y1 = output[1]
                        x2 = output[2]
                        y2 = output[3]


                        if names[c] == 'car' or names[c] == 'truck' or names[c] == 'bus':
                            turn2_count(x1, x2, y2, id)
                            turn3_count(x1, x2, y2, id)
                            turn5_count(x1, x2, y2, id)
                            turn12_count(x1, x2, y2, id)
                            turn8_count(x1, x2, y2, id)
                            turn11_count(x1, x2, y2, id)
                            turn9_count(x1, x2, y2, id)
                            turn6_count(x1, x2, y2, id)










                        if save_txt:
                            # to MOT format
                            #bbox_left = output[0]
                            #bbox_top = output[1]
                            #bbox_w = output[2] - output[0]
                            #bbox_h = output[3] - output[1]

                            x1=output[0]
                            y1=output[1]
                            x2=output[2]
                            y2=output[3]

                            # Write MOT compliant results to file
                            #with open(txt_path, 'a') as f:
                               #f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,
                                                           #bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                            #if(int(id)==90):

                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, x1, y1, x2, y2, -1, -1, -1, -1))  # label format
                                #if names[c] == 'car' or names[c] == 'truck' or names[c] == 'bus':
                                    #f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, x1, y1, x2, y2))

                            with open(csv_path, 'a') as csvfile:
                                if names[c] == 'car' or names[c] == 'truck' or names[c] == 'bus':
                                    writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n', )
                                    row = [frame_idx + 1,id, x1, y1, x2, y2, names[c]]
                                    writer.writerow(row)



                            # cv2.putText(im0, "Turn type 11 count: " + str(all_new_turn11_count), (0, 330),
                            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                            #
                            # cv2.putText(im0, "Turn type 12 count: " + str(all_new_turn12_count), (0, 350),
                            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                            #
                            # cv2.putText(im0, "Turn type 9 count: " + str(all_new_turn9_count), (0, 370),
                            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                            #
                            # cv2.putText(im0, "Turn type 8 count: " + str(all_new_turn8_count), (0, 390),
                            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                            #
                            #
                            # cv2.putText(im0, "Turn type 3 count: " + str(all_new_turn3_count), (0, 410),
                            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                            #
                            # cv2.putText(im0, "Turn type 2 count: " + str(all_new_turn2_count), (0, 430),
                            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                            #
                            # cv2.putText(im0, "Turn type 5 count: " + str(all_new_turn5_count), (0, 450),
                            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

                            #cv2.putText(im0, "Turn type 6 count: " + str(all_new_turn6_count), (0, 470),
                                        #cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)





            else:
                deepsort.increment_ages()

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results

            im0 = annotator.result()
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'


                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h+180))
                im0 = cv2.copyMakeBorder(im0, bottom=180,top=0,left=0,right=0, borderType=cv2.BORDER_CONSTANT, value=0)

                cv2.putText(im0, "Turn 2: " + str(all_new_turn2_count), (0, 500),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                cv2.putText(im0, "Turn 3: " + str(all_new_turn3_count), (0, 520),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                cv2.putText(im0, "Turn 5: " + str(all_new_turn5_count), (0, 540),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                cv2.putText(im0, "Turn 6: " + str(all_new_turn6_count), (0, 560),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                cv2.putText(im0, "Turn 8: " + str(all_new_turn8_count), (0, 580),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                cv2.putText(im0, "Turn 9: " + str(all_new_turn9_count), (0, 600),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                cv2.putText(im0, "Turn 11: " + str(all_new_turn11_count), (0, 620),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                cv2.putText(im0, "Turn 12: " + str(all_new_turn12_count), (0, 640),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                vid_writer.write(im0)
                #w:720, h:480



    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
