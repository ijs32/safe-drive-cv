import torch, torchvision.transforms as T, utils.image_preprocessing as ip
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, crop

class StackModels():
    def __init__(self, object_detection_weights, object_detection_model, classification_model):

        self.object_detection_weights = object_detection_weights
        self.object_detection_model = object_detection_model

        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.classification_model = classification_model
        self.classification_model.to(self.device)

    
    def run_models(self, frame):
        """Runs the object detection and classification models on a frame."""
        bbox, cellphone = self._detect_phone(frame)
        
        # If a cell phone is detected, run the classification model.
        if cellphone:
            allowed = self._classify_phone_use(bbox)
            if not allowed:
                self._alert_user()

    def _detect_phone(self, frame):
        """Detects a cell phone in a frame and returns the cropped image of the cell phone."""
        
        # Apply preprocessing to the frame
        preprocess = self.object_detection_weights.transforms()
        batch = [preprocess(frame)]

        # Draw bounding boxes around the detected objects
        prediction = self.object_detection_model(batch)[0]
        labels = [self.object_detection_weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(frame, boxes=prediction["boxes"],
                                labels=labels,
                                colors="red",
                                width=4)
        
        # Crop the cell phone from the frame
        for prediction, label in zip(prediction["boxes"], labels):
            if label == "cell phone":
                bbox = (prediction).int()
                # Extract the bounding box dimensions
                left, top, right, bottom = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()

                # Calculate the width and height of the bounding box
                bbox_width = right - left
                bbox_height = bottom - top
                cropped_frame = crop(frame, top, left, bbox_height, bbox_width)

                return to_pil_image(cropped_frame), True
            
        return to_pil_image(box.detach()), False

    def _classify_phone_use(self, bbox):
        """Classifies the use of a cell phone."""

        # Preprocess the image
        predicted_bbox = ip.pre_process(bbox)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])
        predicted_bbox = transform(predicted_bbox)
        predicted_bbox = predicted_bbox.to(torch.device('mps')).float().unsqueeze(0)

        # Run the classification model
        outputs = self.classification_model(predicted_bbox)
        return True if outputs.item() < 0.5 else False
    
    def _alert_user(self):
        print("DING DING DING")