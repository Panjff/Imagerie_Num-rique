import subprocess

WORKSPACE_PATH = 'C:/Users/22170/Documents/_project/workspace'
SCRIPTS_PATH = 'C:/Users/2217/Documents/Sign_project/'
APIMODEL_PATH = 'C:/Users/2170/Documents/Sign_/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/Images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/ssd_mobnet/'

commande1 = f"python {SCRIPTS_PATH}/generate_tfrecord.py -j {IMAGE_PATH}/Test -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/test.record"
commande2 = f"python {SCRIPTS_PATH}/generate_tfrecord.py -j {IMAGE_PATH}/Training -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/train.record"

subprocess.run(commande1, shell=True)
subprocess.run(commande2, shell=True)