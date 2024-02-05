import tensorflow as tf
from object_detection.utils import config_util
import os
from object_detection.builders import model_builder
import detection_model


WORKSPACE_PATH = 'C:/Users/22170/Documents/Sign_project/workspace'
SCRIPTS_PATH = 'C:/Users/22170/Documents/Sign_project/codes'
APIMODEL_PATH = 'C:/Users/22170/Documents/Sign_project/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/Images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/ssd_mobnet/'


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections