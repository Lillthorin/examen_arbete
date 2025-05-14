#Import av bibliotek som används

import cv2
import numpy as np
from call_app import predict_no_label
from robot import PersistentRobotClient
from simple_grab import connect_camera, start_grabbing, get_3d_coordinates
import time

#Globala variabler
y_limit = 20
coord = 'COORD'
exclusion_list = []
depth = None
x_mm, y_mm, z_mm = 0, 0, 0

# Upprättar kontakt med robot på ip, socket.
robot = PersistentRobotClient('192.168.125.1', 5000)


print("Attempting to connect with camera...")
camera = connect_camera()
print("Connected to camera...")

def group_and_sort_bbox_by_y_center(bboxes, y_threshold=5):
    """
    Dela upp listan med bboxar i rader baserat på y_centrum och sortera varje rad på x_centrum.

    Args:
        bboxes (list): En lista med bboxar i formatet [xmin, ymin, xmax, ymax, (x_cen, y_cent)].
        y_threshold (int): Maximal skillnad mellan y_cent för att tillhöra samma rad.

    Returns:
        list: En array där varje rad innehåller bboxar sorterade på x_centrum.
    """
    from itertools import groupby
    
    # Sortera bboxarna baserat på y_cent först
    bboxes = sorted(bboxes, key=lambda box: box[-1][1])

    # Gruppindelning baserat på y_cent och y_threshold
    grouped_rows = []
    current_row = [bboxes[0]]

    for i in range(1, len(bboxes)):
        current_box = bboxes[i]
        last_box = current_row[-1]

        # Kolla om skillnaden mellan y_cent är inom y_threshold
        if abs(current_box[-1][1] - last_box[-1][1]) <= y_threshold:
            current_row.append(current_box)
        else:
            # Sortera raden baserat på x_cent och lägg till i grupperade rader
            grouped_rows.append(sorted(current_row, key=lambda box: box[-1][0]))
            current_row = [current_box]

    #Lägger till sista raden
    grouped_rows.append(sorted(current_row, key=lambda box: box[-1][0]))

    return grouped_rows

#Returnerar centrum av bboxen som skickats från modellen
def center(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (int((x_min + x_max) / 2), int((y_min + y_max) / 2))

def is_inside_exclusion(center, exclusion_list):
    # Kontrollera om centrumkoordinaterna är inom någon av bounding boxarna i exclusion_list
    for ex_box in exclusion_list:
        x_min, y_min, x_max, y_max = ex_box
        if x_min <= center[0] <= x_max and y_min <= center[1] <= y_max:
            return True
    return False

# Ta en bild och skicka till predict_no_label som returnerar prediktioner.
def predict(exclusion_list):
    global camera
    img_array, depth = start_grabbing(camera=camera)

    #Felhantering
    try:
        results, img = predict_no_label(img_array)
        if type(results)==int:
            time.sleep(2)
            predict(exclusion_list=exclusion_list)

    except Exception as e:
        print(e)
        time.sleep(2)
        img_array, depth = start_grabbing(camera=camera)
        results, img = predict_no_label(img_array)
        
    detections = []
   
    if results == []:
        return [], [], depth
    for bbox in results:
        x1, y1, x2, y2 = bbox
        box_center = center([x1, y1, x2, y2])
        if not is_inside_exclusion(box_center, exclusion_list):
            detections.append([x1, y1, x2, y2, box_center])    
    
    # Sortera efter y-koordinaten för centrumet
    sorted_detections = sorted(detections, key=lambda x: x[-1][1])
    sorted_detections1 = []
    if sorted_detections:
        grouped = (group_and_sort_bbox_by_y_center(sorted_detections, y_threshold=y_limit))
        for row in grouped:
            for obj in row:
                sorted_detections1.append(obj)
        
        # Ritar detektioner, en cirkel ritas i det paket som ska plockas först.
        if sorted_detections1:
            x_min, y_min, x_max, y_max = map(int, sorted_detections1[0][:4])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=1)
            cv2.circle(img, (sorted_detections1[0][4]), 5, (0, 0, 255), thickness=-1)
            
        for bboxes in sorted_detections1[1:]:
            x_min, y_min, x_max, y_max = map(int, bboxes[:4])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=1)
          
   
    return img, sorted_detections1, depth


# Huvudfunktion
def main():

    global exclusion_list,  camera, depth, x_mm, y_mm, z_mm, img
    #Invänta start
    input('Tryck enter för att börja med nya kartonger')
    #Hämta första bild och koordinater för bbox
    img, sorted_detections, depth = predict(exclusion_list)
    
    
    while True:
        try:
            cv2.imwrite("img.png", img)
        except Exception:
            print('ingen bild att visa')
        #Läser in robotmeddelande och väntar på att få ready
        key = cv2.waitKey(30)
        rob_msg = robot.receive_response()
        print("Robotmeddelande:", rob_msg)
        
        if len(sorted_detections) == 0:
            #Om inga nya detektioner görs vänta på att 'ny pall' ställts fram i detta fall
            #behöver nya kartonger placeras ut.
            exclusion_list = []
            input('Tryck enter för att börja med nya kartonger')
            img, sorted_detections, depth = predict(exclusion_list)

       

        if rob_msg == "READY": 
    
            if len(sorted_detections) != 0:
                x_min, y_min, x_max, y_max = map(int, sorted_detections[0][:4])
                x_mm, _, z_mm = get_3d_coordinates(center=sorted_detections[0][4], depth_map=depth, camera=camera)
                print(f'centrum högst = {(int((x_min + x_max) / 2), int(y_min+3))} Vanligt centrum {sorted_detections[0][4]}')
                _, y_mm, _ = get_3d_coordinates(center=(int((x_min + x_max) / 2), int(y_min+3)), depth_map=depth, camera=camera)
                #Om inga värden från djupkartan anges, ta en ny bild och testa igen. 
                if x_mm == None:
                    img, sorted_detections, depth = predict(exclusion_list)

                rob_msg = ''
                coordinates_string = f"{int(z_mm+140):04d},{int((-1*x_mm)-163):04d},{int(((y_mm*-1)+(70+230))):04d}"
                #skickar "COORD" till robot för att starta rätt program
                robot.sock.sendall(coord.encode('utf-8'))            
                
                time.sleep(0.5)
                #skickar x, y, z koordinater till roboten.
                robot.send_coordinates(coordinates_string)
               
                #Handskakningsfunktion med robot! 
                while True:
                    rob_msg = robot.receive_response()
                    print("Robotmeddelande:", rob_msg)
                    if rob_msg == "DONE":
                        time.sleep(0.1)
                        robot.sock.sendall("OK".encode('utf-8'))  
                        break
                    time.sleep(0.1)
                rob_msg = ''
                #Om det finns fler paket vänta lägg blockera nya detekteringar där senaste paketet plockades
                if len(sorted_detections) != 1:
                    exclusion_list.append(sorted_detections[0][:4])
                #Annars töms listan med blockerande koordinater
                else:
                    exclusion_list = []
                #Ta ny bild 
                img, sorted_detections, depth = predict(exclusion_list)
                
                
        
        elif key == 27:  # Escape-nyckel
            break
    

if __name__ == "__main__":
    main()