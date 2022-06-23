import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import spatial
from collections import Counter
import os
from imageai.Detection import ObjectDetection
import time
from datetime import datetime
import glob
import time
from shapely.geometry import LineString, Polygon
from descartes import PolygonPatch
# import DOTCameraCollector


execution_path = os.getcwd()
# color_index = {'person': 'blue', 'bus': 'maroon', 'car': 'orange', 'truck': 'indigo', 'motorcycle': 'azure',
#                'parking meter': 'aliceblue', 'bicycle': 'olivedrab', 'umbrella': 'deeppink',
#                'traffic light': 'mediumblue', 'backpack': 'slategray',  'stop sign': 'beige'}

color_index = {'person': 'maroon', 'bus': 'mediumblue', 'car': 'orange', 'truck': 'indigo', 'motorcycle': 'azure',
               'parking meter': 'aliceblue', 'bicycle': 'olivedrab', 'umbrella': 'deeppink',
               'traffic light': 'mediumblue', 'backpack': 'slategray',  'stop sign': 'beige'}

mpl.rcParams['font.size'] = 22.0
#mpl.rcParams['figure.constrained_layout.use'] = True
# file_name = "Main St @ Roosevelt Ave_20200513-161045"
# file_suf = ".jpg"
# frame_array = [{'box_points': [245, 103, 267, 129], 'percentage_probability': 51.974618434906006, 'name': 'bus'}, {'box_points': [298, 120, 340, 191], 'percentage_probability': 92.36682653427124, 'name': 'bus'}, {'box_points': [199, 110, 248, 153], 'percentage_probability': 98.27568531036377, 'name': 'bus'}, {'box_points': [306, 118, 337, 144], 'percentage_probability': 35.03006398677826, 'name': 'car'}, {'box_points': [273, 111, 285, 124], 'percentage_probability': 66.9325590133667, 'name': 'car'}, {'box_points': [131, 147, 138, 165], 'percentage_probability': 36.2542450428009, 'name': 'person'}, {'box_points': [102, 156, 111, 171], 'percentage_probability': 39.99610245227814, 'name': 'person'}, {'box_points': [196, 127, 201, 144], 'percentage_probability': 41.54825508594513, 'name': 'person'}, {'box_points': [108, 157, 115, 171], 'percentage_probability': 43.33010017871857, 'name': 'person'}, {'box_points': [87, 177, 96, 207], 'percentage_probability': 47.5646436214447, 'name': 'person'}, {'box_points': [120, 152, 131, 172], 'percentage_probability': 69.66400146484375, 'name': 'person'}, {'box_points': [36, 189, 48, 225], 'percentage_probability': 85.5470359325409, 'name': 'person'}, {'box_points': [141, 193, 154, 230], 'percentage_probability': 92.09975600242615, 'name': 'person'}, {'box_points': [177, 131, 182, 146], 'percentage_probability': 96.31378650665283, 'name': 'person'}, {'box_points': [133, 165, 144, 192], 'percentage_probability': 98.99353981018066, 'name': 'person'}]
# frame_array = [{'name': 'bus', 'box_points': [210, 106, 252, 149], 'percentage_probability': 94.21907663345337}, {'name': 'car', 'box_points': [207, 153, 251, 191], 'percentage_probability': 97.61119484901428}, {'name': 'person', 'box_points': [113, 157, 121, 175], 'percentage_probability': 35.31881868839264}, {'name': 'person', 'box_points': [169, 139, 175, 150], 'percentage_probability': 40.51617681980133}, {'name': 'person', 'box_points': [25, 171, 32, 195], 'percentage_probability': 43.7575101852417}, {'name': 'person', 'box_points': [147, 138, 158, 162], 'percentage_probability': 47.45425581932068}, {'name': 'person', 'box_points': [41, 184, 50, 210], 'percentage_probability': 73.7103819847107}, {'name': 'person', 'box_points': [168, 139, 177, 160], 'percentage_probability': 78.87740731239319}, {'name': 'person', 'box_points': [98, 169, 109, 197], 'percentage_probability': 84.46545004844666}, {'name': 'person', 'box_points': [56, 198, 68, 232], 'percentage_probability': 88.51633071899414}, {'name': 'person', 'box_points': [19, 201, 32, 239], 'percentage_probability': 88.5682761669159}, {'name': 'person', 'box_points': [140, 159, 150, 184], 'percentage_probability': 91.33086204528809}, {'name': 'person', 'box_points': [177, 136, 184, 153], 'percentage_probability': 95.34892439842224}, {'name': 'person', 'box_points': [38, 208, 50, 240], 'percentage_probability': 97.74278402328491}, {'name': 'person', 'box_points': [127, 165, 139, 192], 'percentage_probability': 98.70615005493164}]

resized = False
# record = []

def threshold(distance, height, p, q):
    return distance[p, q]/((height[p]+height[q])/2)


def frame_processing(file_name, frame_array):
    this_colors = []
    labels = []
    sizes = []
    box = []
    items = []
    height = []
    bottom = []
    centroids = []
    counter = 0

    # polygon_5_49_1 = Polygon([(107, 72), (130, 70), (195, 162), (120, 174)])
    # polygon_5_49_2 = Polygon([(130, 70), (146, 68), (226, 134), (179, 139)])
    # polygon_5_49_4 = Polygon([(165, 66), (181, 66), (310, 122), (270, 127)])
    # polygon_5_49_5 = Polygon([(181, 65), (198, 63), (344, 117), (302, 123)])
    # polygon_5_49_p = Polygon([(157, 145), (331, 119), (336, 157), (149, 200)])
    # polygon_5_49_r = Polygon([(148, 69), (166, 65), (266, 127), (224, 132)])
    # polygon_5_42_1 = Polygon([(70, 48), (86, 46), (72, 126), (24, 132)])
    # polygon_5_42_2 = Polygon([(81, 51), (96, 51), (119, 113), (73, 122)])
    # polygon_5_42_3 = Polygon([(98, 51), (114, 49), (168, 108), (118, 117)])
    # polygon_5_42_4 = Polygon([(114, 50), (128, 49), (212, 108), (168, 110)])
    # polygon_5_42_5 = Polygon([(126, 49), (147, 46), (253, 102), (209, 108)])

    plt.clf()
    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(width=2000, height=400)
        resized = True

    current_image=img.imread(file_name)

    plt.subplot(1, 1, 1)
    plt.title("Test", loc='left')
    plt.axis("off")
    plt.imshow(current_image, interpolation="none")

    # study_area_1 = PolygonPatch(polygon_5_49_1, color = 'r', alpha = 0.3)
    # study_area_5 = PolygonPatch(polygon_5_49_5, color = 'c', alpha = 0.3)
    # plt.gca().add_patch(study_area_1)
    # plt.gca().add_patch(study_area_5)
    # study_area_1 = PolygonPatch(polygon_5_42_1, color = 'r', alpha = 0.3)
    # study_area_2 = PolygonPatch(polygon_5_42_5, color = 'c', alpha = 0.3)
    # plt.gca().add_patch(study_area_1)
    # plt.gca().add_patch(study_area_2)

    for eachObject in frame_array:
        items.append(eachObject['name'])
        if eachObject['name'] == 'person':
            box.append(eachObject)
            rect = plt.Rectangle((eachObject['box_points'][0], eachObject['box_points'][1]),
                                 (eachObject['box_points'][2]-eachObject['box_points'][0]), (eachObject['box_points'][3]-eachObject['box_points'][1]),
                                 linewidth=2, edgecolor=color_index[eachObject['name']], facecolor=color_index[eachObject['name']], alpha = 0.5)
        else:
            rect = plt.Rectangle((eachObject['box_points'][0], eachObject['box_points'][1]),
                                     (eachObject['box_points'][2]-eachObject['box_points'][0]), (eachObject['box_points'][3]-eachObject['box_points'][1]),
                                     linewidth=2, edgecolor=color_index[eachObject['name']], facecolor='None')
        plt.gca().add_patch(rect)
        plt.gca().text(eachObject['box_points'][0]+0.6, eachObject['box_points'][1]-1.3,
                       '{:s} {:.2f}'.format(eachObject['name'], eachObject['percentage_probability']),
                       bbox=dict(boxstyle='round', facecolor=color_index[eachObject['name']], alpha=0.5), fontsize=22, color='white')

    item_count = Counter(items)

    for eachItem in item_count:
        labels.append(eachItem + " = " + str(item_count[eachItem]))
        sizes.append(item_count[eachItem])
        this_colors.append(color_index[eachItem])
    # print (labels, sizes)
    item_count['time'] = [file_name]
    # print ("item_count:", item_count)

    if len(box) != 0:
        for j in box:
            coord = j['box_points']
            centerCoord = (coord[0] + ((coord[2] - coord[0]) / 2), coord[1] + ((coord[3] - coord[1]) / 2))
            centroids.append(centerCoord)
            height.append(coord[3] - coord[1])
            bottom.append(coord[3])

        tree = spatial.KDTree(centroids)
        t_dst = tree.sparse_distance_matrix(tree, 500)
        t_dst = t_dst.toarray()
        t_dst = np.array(t_dst, dtype=np.int32)

        for p in range(len(t_dst)):
            for q in range(len(t_dst)):
                if threshold(t_dst, height, p, q) <= 1.045 and t_dst[p, q] != 0:
                    counter += 1
                    x_values = []
                    y_values = []
                    x_values.append(centroids[p][0])
                    x_values.append(centroids[q][0])
                    y_values.append(centroids[p][1])
                    y_values.append(centroids[q][1])
                    plt.plot(x_values, y_values, 'c-', linewidth=6, alpha=0.75)
                    plt.text((centroids[p][0] + centroids[q][0]) / 2, (centroids[p][1] + centroids[q][1]) / 2,
                             str("{:.2f}".format(3.28 * 1.7 * t_dst[p, q] / ((height[p] + height[q]) / 2))), fontsize=10,
                             color='white')

    colors = ['#ff6666', '#ffcc99', '#99ff99', '#66b3ff', '#c2c2f0']

    # plt.subplot(1, 2, 2)
    # plt.pie(sizes, labels=labels, startangle=90, colors=colors, pctdistance=0.6, autopct="%1.1f%%", radius=0.7,
    #         textprops={'fontsize': 18})
    # plt.text(-1.1, 0.85, "Analysis: " + str(int(counter / 2)) + " social distance violations detected.", fontsize=26)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0.1)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gcf().set_size_inches(20, 10)
    plt.savefig(file_name[:-4] + "_masked.jpg", bbox_inches = 'tight', pad_inches = 0,dpi = 100)
    print ("Finishing file:", file_name)
    return item_count
    # plt.pause(0.01)
    # plt.show()

#DOTCameraCollector()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
#detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "..\yolo.h5"))
#detector.setModelPath(os.path.join(execution_path , "..\\resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom = detector.CustomObjects(person=True, car=True, truck=True, bicycle=True, bus=True, motorcycle=True)
# filenames = ["Main St @ Roosevelt Ave_20200513-160915", "Main St @ Roosevelt Ave_20200513-160945", "Main St @ Roosevelt Ave_20200513-161015", "Main St @ Roosevelt Ave_20200513-161045"]
# Record = pd.DataFrame(columns=['time', 'person', 'car', 'truck', 'bicycle', 'bus'])
# for file in filenames:
# while True:
#     with open('listfile.txt', 'r') as filehandle:
#         for line in filehandle:
#             file = line
#     filehandle.close()
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     currentDate = timestamp[:8]

#for file in glob.glob('Images4Processing/test/*'):
for file in glob.glob('Images4Processing/Main St @ Roosevelt Ave_20200513-161045.jpg'):
    program_st = time.time()
    #print("*******************  Start Program  *******************")
    print("Start time %s" % (time.strftime('%X', time.localtime(program_st))))
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=file,
                                                        output_image_path=os.path.join(execution_path , "result_store.jpg"),
                                                        minimum_percentage_probability=45)
    frame_processing(file, detections)
    ed_time = time.time()
    print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
    #print("*******************  End Program  *******************")
    #time.sleep(30)
    # frame_result = frame_processing(file, detections)
    # Record.append(frame_result, ignore_index=True)
    # if len(Record) > 2880:
        # Record.to_csv("Result%n")
#Record.to_csv("Results.csv")

