#librairy
import cv2
import os
import time

WORKSPACE_PATH = '/Sign_project/workspace'
SCRIPTS_PATH = '/Sign_project/codes'
APIMODEL_PATH = '/Sign_/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/Images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/ssd_mobnet/'

#Noms des signes et lettres choisies
labels = [
		  'A',
          'B',
          'C',
          'D',
          'E',
		  'Bonjour',
          'Non',
          'Oui',
          'CaVa',
          'Pardon'
        ]

#Nombre d'images par label
number_imgs = 10

#Enregistrement des images des labels
for label in labels:
    label_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(label_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    print(f'Images pour {label}\n')
    time.sleep(5)
    
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        imagename = os.path.join(label_path, f'{label}_{imgnum}.jpg')
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(5)
        print('Suivant\n\r')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()