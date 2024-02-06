WORKSPACE_PATH = '/Sign_project/workspace'
SCRIPTS_PATH = '/Sign_project/codes'
APIMODEL_PATH = '/Sign_/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/Images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/ssd_mobnet/'


labels = [{'name':'A', 'id':1}, 
          {'name':'B', 'id':2},
          {'name':'C', 'id':3},
          {'name':'D', 'id':4},
          {'name':'E', 'id':5},
          {'name':'Bonjour', 'id':6},
          {'name':'Non', 'id':7},
          {'name':'Oui', 'id':8},
          {'name':'CaVa', 'id':9},
          {'name':'Pardon', 'id':10}
          ]

with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
        
