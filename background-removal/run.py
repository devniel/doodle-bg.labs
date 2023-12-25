from pymatting import cutout
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# Function to apply DeepLabv3 model and get the segmentation mask
def apply_deeplab(image_path, model):
    # Load the image and convert to RGB
    input_image = Image.open(image_path).convert("RGB")
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Check if a GPU is available and if not, use a CPU
    if torch.cuda.is_available():
        print("👽 CUDA mode")
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Apply the model
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)


    # Convert to numpy array
    return output_predictions.byte().cpu().numpy()


def run(image_name, extension="jpg"):
    # Prepare inputs
    input_path = f"./assets/{image_name}.{extension}"
    input_mask_path = f"./output/{image_name}-mask.png"
    input_trimap_path = f"./output/{image_name}-trimap.png"
    output_path = f"./output/{image_name}.png"

    # Load the pre-trained DeepLabv3 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    # Apply the model to your image
    mask = apply_deeplab(input_path, model)
    mask_saved = cv2.imwrite(input_mask_path, mask * 255)

    if(mask_saved):
        print("📝 Mask saved")

    mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

    trimap = mask.copy()
    trimap = trimap.astype("uint8")

    # Erode and dilate the mask
    k_size=(8,8)
    iterations=5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)

    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    cv2.imwrite(f"./output/{image_name}-mask_eroded.png", eroded)
    cv2.imwrite(f"./output/{image_name}-mask_dilated.png", dilated)

    trimap_eroded = np.full(mask.shape, 128)
    trimap_eroded[eroded >= 2] = 255
    cv2.imwrite(f"./output/{image_name}-trimap_eroded.png", trimap_eroded)

    trimap_dilated = np.full(mask.shape, 128)
    trimap_dilated[dilated <= 1] = 0
    cv2.imwrite(f"./output/{image_name}-trimap_dilated.png", trimap_dilated)

    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 1] = 255
    trimap[dilated == 0] = 0

    # Save the trimap
    #cv2.imwrite("./output/segmentation_mask.png", mask * 255)
    saved = cv2.imwrite(input_trimap_path, trimap)

    if(saved):
        print("📝 Trimap saved")
    # Apply background matting
    cutout(input_path, input_trimap_path, output_path)

if __name__ == "__main__":
    run("devniel-avatar")
    #cutout("./assets/lemur.png", "./assets/lemur_trimap.png", "./output/lemur.png")
    #cutout("./assets/devniel.png", "./output/devniel-trimap-2.png", "./output/devniel.png")