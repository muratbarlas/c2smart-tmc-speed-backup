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
from descartes import PolygonPatch
from shapely.geometry import LineString, Polygon
# import DOTCameraCollector


execution_path = os.getcwd()
color_index = {'person': 'maroon', 'bus': 'steelblue', 'car': 'orange', 'truck': 'indigo', 'motorcycle': 'azure',
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
    
    # Define customized polygons You need to finetune them to accurately fitting the area
    polygon_5_42_1 = Polygon([(70, 48), (86, 46), (72, 126), (24, 132)])
    polygon_5_42_2 = Polygon([(81, 51), (96, 51), (119, 113), (73, 122)])
    polygon_5_42_3 = Polygon([(98, 51), (114, 49), (168, 108), (118, 117)])
    polygon_5_42_4 = Polygon([(114, 50), (128, 49), (212, 108), (168, 110)])
    polygon_5_42_5 = Polygon([(126, 49), (147, 46), (253, 102), (209, 108)])

    plt.clf()
    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(width=2000, height=400)
        resized = True

    current_image=img.imread(file_name)

    plt.subplot(1, 1, 1)
    #plt.title("Test", loc='left')
    plt.axis("off")
    plt.imshow(current_image, interpolation="none")

    # Add customized polygon to the image
    study_area_1 = PolygonPatch(polygon_5_42_1, color = 'r', alpha = 0.3)
    study_area_2 = PolygonPatch(polygon_5_42_5, color = 'c', alpha = 0.3)
    plt.gca().add_patch(study_area_1)
    plt.gca().add_patch(study_area_2)

    for eachObject in frame_array:
        items.append(eachObject['name'])
        # if eachObject['name'] == 'person':
        #     box.append(eachObject)

        # Get the bottom line of a bounding box
        the_line = LineString([(eachObject['box_points'][0],eachObject['box_points'][3]),(eachObject['box_points'][2],eachObject['box_points'][3])])
        
        # Obtain the overlapped part of the bottom line of bounding box and the study area
        length_inter_1 = the_line.intersection(polygon_5_42_1).length
        length_inter_5 = the_line.intersection(polygon_5_42_5).length
        
        # Get the length of the bottom line
        length_full = the_line.length

        # Using the ratio of overlapped part and the whole bottom line to decide some processing
        if (length_inter_1/length_full >=0.5) and (eachObject['name'] != "person") and (eachObject['name'] != "bus"):
            #print ('Yeah')
            rect = plt.Rectangle((eachObject['box_points'][0], eachObject['box_points'][1]),
                                 (eachObject['box_points'][2]-eachObject['box_points'][0]), (eachObject['box_points'][3]-eachObject['box_points'][1]),
                                 linewidth=3, edgecolor=color_index[eachObject['name']], facecolor='none')
            plt.gca().add_patch(rect)
            plt.gca().text(eachObject['box_points'][0]+0.6, eachObject['box_points'][1]-1.3,
                           '{:s}'.format("Illegal Parking"),
                           bbox=dict(boxstyle='round', facecolor=color_index[eachObject['name']], alpha=0.7), fontsize=12, color='white')
            #print (data)
        elif (length_inter_5/length_full >=0.3) and (eachObject['name'] != "person") and (eachObject['name'] != "bus"):
            #print ('Yeah')
            rect = plt.Rectangle((eachObject['box_points'][0], eachObject['box_points'][1]),
                                 (eachObject['box_points'][2]-eachObject['box_points'][0]), (eachObject['box_points'][3]-eachObject['box_points'][1]),
                                 linewidth=3, edgecolor=color_index[eachObject['name']], facecolor='none')
            plt.gca().add_patch(rect)
            plt.gca().text(eachObject['box_points'][0]+0.6, eachObject['box_points'][1]-1.3,
                           '{:s}'.format("Illegal Parking"),
                           bbox=dict(boxstyle='round', facecolor=color_index[eachObject['name']], alpha=0.7), fontsize=12, color='white')