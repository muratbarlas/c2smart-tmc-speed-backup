#speed graph

import cv2
import torch
import torch.backends.cudnn as cudnn
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import csv
import pandas as pd

cars = {}


def calculate_speed(start_frame, end_frame):
    diff = end_frame - start_frame
    dist = 0.014039773 #this is in miles, it's equal to 74.13 feet
    time = (diff/30)/3600 #hour
    speed = dist/time #miles/hour
    return speed

class Speed:
    def __init__(self, speed, frame):
        self.speed = speed
        self.frame = frame


class Car:
    def __init__(self, id):
        self.id = id
        self.start_frame = None
        self.end_frame = None
        self.closed = False

    def return_start(self):
        return self.start_frame

    def return_end(self):
        return self.end_frame

    def closed_(self):
        self.closed = True


#frame, vehicle_id, x1, y1, x2, y2, classification

rect1= [250, 220, 350, 280]
rect2 = [1000, 300,1150, 460]

def convertToPoly(objDims):  # [x1, y1, x2, y2]
    box = objDims
    b1_x1, b1_y1, b1_x2, b1_y2 = box[0], box[1], box[2], box[3]
    poly = Polygon([(b1_x1, b1_y1), (b1_x2, b1_y1), (b1_x2, b1_y2), (b1_x1, b1_y2), (b1_x1, b1_y1)])
    return poly



rect1_poly = convertToPoly(rect1)
rect2_poly = convertToPoly(rect2)






print('started')

def zone_check(x1, x2, y2, zone):
    the_line = LineString([(x1, y2), (x2, y2)])
    if zone==1:
        length_intersect = the_line.intersection(rect1_poly).length
    elif zone==2:
        length_intersect = the_line.intersection(rect2_poly).length
    length_full = the_line.length
    if length_intersect / length_full > 0.3:
        return True



with open('speed_long_formatted.csv', 'r') as read_obj:
#with open('full_check.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj, delimiter='\t', lineterminator='\n', )
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        #print(row)
        frame_idx = int(row[0])
        x1 = int(row[2])
        y1= int(row[3])
        x2= int(row[4])
        y2=int(row[5])
        id = row[1]
        vehicle_type = row[6]

        if vehicle_type == 'car' or vehicle_type == 'truck' or vehicle_type == 'bus':
            if id not in cars:
                vehicle = Car(id)
            else:
                vehicle = cars[id]

            if vehicle.closed == False:
                if zone_check(x1, x2, y2, 1) == True and vehicle.start_frame == None:
                    vehicle.start_frame = frame_idx
                    cars[id] = vehicle

                elif zone_check(x1, x2, y2, 2) == True and vehicle.end_frame == None:
                    vehicle.end_frame = frame_idx
                    vehicle.closed_()
                    cars[id] = vehicle


    print('ending')
print('ended')


list_all_speeds = []
exit_frames = []



df_speeds = pd.DataFrame(columns=['Frame', 'Speed'])



for value in cars.values():
    if value.start_frame != None and value.end_frame != None:
        #print('id:', value.id, ' start:', value.start_frame, ' end:', value.end_frame)
        speed = calculate_speed(value.start_frame, value.end_frame)
        list_all_speeds.append(speed)
        exit_frames.append(value.end_frame)
        df_speeds = df_speeds.append({'Frame': int(value.end_frame), 'Speed': int(speed)}, ignore_index=True)




#KEEP--------------
#print('mean speed: ',sum(list_all_speeds) / len(list_all_speeds))
#plt.scatter(exit_frames,list_all_speeds)
#plt.xlabel("Exit frame")
#plt.ylabel("Speed (mph)")
#plt.show()
#-----------------------

list_minute_frames = []
for i in range (1,31):
    list_minute_frames.append(1800*i)

# list_minute_frames=[1800, 3600, 5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600, 23400, 25200, 27000, 28800, 30600, 32400, 34200, 36000, 37800, 39600, 41400, 43200, 45000, 46800, 48600, 50400, 52200, 54000]
#print(list_minute_frames)

df_speeds.sort_values('Frame',inplace=True, ascending=True)

#df_speeds.to_csv('out_csv.csv')


dict_minutes = {}

for i in range(len(list_minute_frames)):
    df_holder = None
    if (i == 0):
        df_holder = df_speeds[df_speeds['Frame'].between(0, list_minute_frames[0])]
        print(df_holder)

    elif (i !=0):
        df_holder = df_speeds[df_speeds['Frame'].between(list_minute_frames[i-1], list_minute_frames[i])]


    dict_minutes[i+1] = df_holder["Speed"].mean()

print(dict_minutes)





plt.scatter(dict_minutes.keys(), dict_minutes.values())
plt.xlabel("Minutes")
plt.ylabel("Average speed(mph)")
plt.xticks(list(dict_minutes.keys()))
plt.show()