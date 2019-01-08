#face recognition from camera
import dlib         
import numpy as np  
import cv2         
import pandas as pd 
import os


# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# calulation disntance between vector and vector
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("e_distance: ", dist)

    if dist > 0.3:
        return "diff"
    else:
        return "same"


# save the face feature
path_features_known_csv = "data/features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)

# the array used for save face feature
features_known_arr = []


# known faces
for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.ix[i, :])):
        features_someone_arr.append(csv_rd.ix[i, :][j])
    features_known_arr.append(features_someone_arr)
print("Faces numbers in Database：", len(features_known_arr))

# face detection and landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_5_face_landmarks.dat')

# opencv camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(3, 640)


while cap.isOpened():

    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)
   
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)    
    faces = detector(img_gray, 0)

    
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)

    

    # face name and position
    pos_namelist = []
    name_namelist = []
    name_name = os.listdir('data/data_csvs_from_camera')
    for i in range(len(name_name)):
        name_name[i] = (name_name[i].split('.')[0])
        
    #print(name_name)
   
    if kk == ord('q'):
        break
    else:        
        if len(faces) != 0:           
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))
          
            for k in range(len(faces)):              
                name_namelist.append("unknown")                
                pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                #compare
                for i in range(len(features_known_arr)):                
                    compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                    if compare == "same":  
                        name_namelist[k] = name_name[i]
    
                
                for kk, d in enumerate(faces):                    
                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

            for i in range(len(faces)):
                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("camera", img_rd)


cap.release()
cv2.destroyAllWindows()
