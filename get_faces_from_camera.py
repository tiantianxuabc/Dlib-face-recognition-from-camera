import dlib
import numpy as np
import cv2 

import os
import shutil

# face detection
detector = dlib.get_frontal_face_detector()

# landmark
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
#cap.set(3, 640)

#the number of people
cnt_ss = 0
#file path for save
current_face_dir = 0
path_make_dir = "data/data_faces_from_camera/"
path_csv = "data/data_csvs_from_camera/"


# dell all file that have saved
def pre_work():    
    if os.path.isdir(path_make_dir):
        pass
    else:
        os.mkdir(path_make_dir)
    if os.path.isdir(path_csv):
        pass
    else:
        os.mkdir(path_csv)    
    folders_rd = os.listdir(path_make_dir)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_make_dir+folders_rd[i])

    csv_rd = os.listdir(path_csv)
    for i in range(len(csv_rd)):
        os.remove(path_csv+csv_rd[i])
        
#dell all file
#pre_work()


person_cnt = 0

# The flag of if u can save images
save_flag = 1

while cap.isOpened():
    # 480 height * 640 width
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    
    # face detection
    faces = detector(img_gray, 0)
   
    font = cv2.FONT_HERSHEY_COMPLEX

    #enter 'n' for save face
    if kk == ord('n') or kk == ord('N'):
        person_cnt += 1
        print("input you name")
        name = input("input you name: \n") 
        if name.strip() != '':
            current_face_dir = path_make_dir + name
        else: 
            current_face_dir = path_make_dir + "person_" + str(person_cnt)
        #print(current_face_dir)
        for dirs in (os.listdir(path_make_dir)):
            if current_face_dir == path_make_dir + dirs:
                shutil.rmtree(current_face_dir)
                print("file has build:", current_face_dir)
        os.makedirs(current_face_dir)
        print("the new face path is: ", current_face_dir)


        cnt_ss = 0

    if len(faces) != 0:      
        for k, d in enumerate(faces):
            
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])
           
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height/2)
            ww = int(width/2)
           
            color_rectangle = (255, 255, 255)
            if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                save_flag = 0
            else:
                color_rectangle = (255, 255, 255)
                save_flag = 1

            cv2.rectangle(img_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)
           
            im_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

            if save_flag:                
                if kk == ord('s'):
                    cnt_ss += 1
                    for ii in range(height*2):
                        for jj in range(width*2):
                            im_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                    cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_ss) + ".jpg", im_blank)
                    print("save：", str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
                    #cnt_ss += 1
                    #cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_ss) + ".jpg", cv2.flip(im_blank, 1))
                    #print("save：", str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")

       
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

   
    cv2.putText(img_rd, "Face Register", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "N: New face folder", (20, 350), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "S: Save face", (20, 400), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    if kk == ord('q') or kk == ord('Q'):
        break
    
    cv2.imshow("camera", img_rd)


cap.release()
cv2.destroyAllWindows()