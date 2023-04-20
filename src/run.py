from stack_models import StackModels
from classification_model import NSASpyware
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import cv2, torch

# Load pretrained faster rcnn model
object_detection_weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
object_detection_model = fasterrcnn_mobilenet_v3_large_fpn(weights=object_detection_weights, box_score_thresh=0.85)
object_detection_model.eval()

# Load pretrained 2D CNN classification model
classification_model = NSASpyware()
classification_model.load_state_dict(torch.load("./saved_models/4_model_9615_20230417.pt"))
classification_model.eval()
classification_model.to(torch.device('mps'))

# Initialize the stack models class
stacked_models = StackModels(object_detection_weights, object_detection_model, classification_model)

# Run the models on a video stream
vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
prediction = None

while success:

    # RCNN expects a tensor of shape (C, H, W)
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_cv2)
    tensor = tensor.permute(2, 0, 1)

    stacked_models.run_models(tensor)
    
    success,image = vidcap.read()