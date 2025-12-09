from ultralytics import YOLO

import torch
from torchvision import transforms

import cv2
import PIL

# test image
image_path = 'inference/test_images/bad_soil2.jpg'

# yolo params
confidence = 0.3

# efficientnet params
input_size = 224
mean = [0.6368, 0.7232, 0.5855]
std = [0.3056, 0.2899, 0.3960]
class_num = 9
class_names = ['Anthracnose', 'Bacterial Blight', 'Citrus Canker', 'Curl Virus', 'Deficiency Leaf', 'Dry Leaf', 'Healthy Leaf', 'Sooty Mould', 'Spider Mites']

# set up torch to use either cuda or cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
	
# load yolo model
yolo_model = YOLO('inference/yolo100epochs.pt')

# load efficientnet model
efficientnet_model = torch.load('inference/efficientnet50epochs.pth', weights_only=False, map_location=device)

# compose transform
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])
	
# set up results list
preds = []

# run yolo model and get results
results = yolo_model.predict(image_path, conf=confidence, verbose=False)

# iterate over results and get efficientnet results
img = cv2.imread(image_path)
for i, box in enumerate(results[0].boxes.xyxy):
    # convert to PIL image
    x1, y1, x2, y2 = map(int, box[:4])
    cropped_image = img[y1:y2, x1:x2]
    pilimg = PIL.Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    
    # transform and unsqueeze for model input
    readyimg = test_transform(pilimg)
    readyimg = readyimg.unsqueeze(0)
    
	# move ready image to device and run efficientnet model
    readyimg = readyimg.to(device)
    with torch.no_grad():
        efficientnet_model.eval()
        outputs = efficientnet_model(readyimg)
        
    # save results
    preds.append(outputs.data)
	
# results saved into preds array
print(preds)