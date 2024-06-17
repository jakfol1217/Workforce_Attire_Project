from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
import yolov7
import os
import json
import sys
import random
from haishoku.haishoku import Haishoku # for detecting the dominant color
from PIL import Image
import numpy as np
from rembg import remove # for removing background

def human_detection(human_detector, img_path, img_pad=None):
    # perform inference
    results = human_detector(img_path)

    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2

    detections = []
    if img_pad is None:
        img_pad = 0
    for bbox in boxes:
        x_tl, y_tl, x_br, y_br = bbox
        x_tl = int(x_tl.item() - img_pad)
        y_tl = int(y_tl.item() - img_pad)
        x_br = int(x_br.item() + img_pad)
        y_br = int(y_br.item() + img_pad)
        #detections.append([x_tl, y_tl, x_br - x_tl, y_br - y_tl]) # xywh
        detections.append([x_tl, y_tl, x_br, y_br]) # xyxy
    return detections


def fix_channels(t):
    """
    Some images may have 4 channels (transparent images) or just 1 channel (black and white images), in order to let the images have only 3 channels. I am going to remove the fourth channel in transparent images and stack the single channel in back and white images.
    :param t: Tensor-like image
    :return: Tensor-like image with three channels
    """
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)

cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

def idx_to_text(i):
    return cats[i]

# Random colors used for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, image_path):
    fig = plt.figure(
        figsize=(16,10)
    )
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        ax.text(xmin, ymin, idx_to_text(cl), fontsize=10,
                bbox=dict(facecolor=c, alpha=0.8))
    plt.axis('off')
    #plt.show()
    plt.savefig(image_path)
    fig.canvas.draw()
    return Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

def visualize_predictions(image, outputs, image_path, threshold=0.8):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    # plot results
    return plot_results(image, probas[keep], bboxes_scaled, image_path)

def per_human_predictor(human_detector, clothing_detector, feature_extractor, image,
                        img_pad=None, threshold=0.8):
    human_detections = human_detection(human_detector, image, img_pad)
    results = []
    for hd in human_detections:
        cropped_img = image.crop(hd)
        inputs = feature_extractor(images=cropped_img,
                                   return_tensors="pt")  # .to("cuda") # add this if using cuda
        outputs = clothing_detector(**inputs)
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(),
                                       cropped_img.size)
        boxes = bboxes_scaled.detach().numpy()

        elements = []
        for p, bbox in zip(probas[keep], boxes):
            single_element = {}
            single_element["object_name"] = idx_to_text(p.argmax())
            single_element["probability"] = float(p.max())
            single_element["relative_bbox"] = bbox
            absolute_bbox = []
            for i in range(2):
                absolute_bbox.append(bbox[i] + hd[i])
            absolute_bbox.append(bbox[2] + hd[0])
            absolute_bbox.append(bbox[3] + hd[1])
            single_element["absolute_bbox"] = absolute_bbox
            elements.append(single_element)
        single_human_elements = {"human_bbox": hd, "elements": elements}
        results.append(single_human_elements)
    return results

def process_predictions(probas, bboxes, x_tl, y_tl):
  detections = []
  for p, bbox in zip(probas, bboxes):
    cat_id = p.detach().cpu().numpy().argmax()
    x1, y1, x2, y2= bbox.detach().cpu().numpy()
    x1 += x_tl
    x2 += x_tl
    y1 += y_tl
    y2 += y_tl
    detections.append([cat_id,x1,y1,x2,y2])
  return detections

def predict(human_detector, clothes_detector, processor, img_path, img_pad, device="cuda:0"):
  # perform human detection 
  results = human_detector(img_path)

  predictions = results.pred[0]
  boxes = predictions[:, :4] # x1, y1, x2, y2
  scores = predictions[:, 4]
  categories = predictions[:, 5]

  img = Image.open(open(img_path, "rb"))
  img = fix_channels(ToTensor()(img))

  # sometimes pieces of garderob (ex. bag) can be outside the human bbox
  # in this case we may pad the human bbox for it to be used in clothing detection
  if type(img_pad) == int:
    img_pad = (img_pad, img_pad, img_pad, img_pad)
  elif len(img_pad) == 2:
    hor, ver = img_pad
    img_pad = (hor, ver, hor, ver)

  results = []
  # for each human in the image
  for i, bbox in enumerate(boxes):
    record = dict() 
    record["person_id"] = i
    # retrieve top-left and bottom-right coordinates
    x_tl, y_tl = bbox[0].item(), bbox[1].item()
    x_br, y_br = bbox[2].item(), bbox[3].item()
    record["bbox"] = [x_tl, y_tl ,x_br ,y_br]
    
    # perform padding to include pieces of clothing that may be outside the initial bbox
    x_tl = max(0,int(x_tl - img_pad[0]))
    y_tl = max(0,int(y_tl - img_pad[1]))
    x_br = min(int(x_br + img_pad[2]), img.size[0])
    y_br = min(int(y_br + img_pad[3]), img.size[1])
    # get the final human crop
    human_crop = [x_tl, y_tl, x_br, y_br]
    
    img_crop = img.crop(human_crop)
    # run clothing detection for a single person
    inputs = processor(images=img_crop, return_tensors="pt").to(device)
    outputs = clothes_detector(**inputs)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    # keep predicitions with confidence above the threshold 
    keep = probas.max(-1).values > 0.8

    # rescale clothing bboxes from [0..1]x[0..1] to [0..img_crop.w]x[0..img_crop.h]  
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), img_crop.size)
    probas = probas[keep]

    # shift clothing bboxes from [0..img_crop.w]x[0..img_crop.h] to [0..img.w]x[0..img.h] 
    detections = process_predictions(probas, bboxes_scaled, x_tl, y_tl)
    record["clothes"] = []
    for det in detections:
      cat_id = int(det[0])
      bbox = [float(v) for v in det[1:]]
      record["clothes"].append({"category_id": cat_id, "bbox": bbox})

    results.append(record)
  results = {"detections": results}
  return results, img


TRANSPARENCY_LIMIT=125
BACKGROUND_PERCENTAGE_THRESHOLD=0.8
        
COLOR_DICT = { # some basic color RGBs
    "lightgreen":np.array([0, 255, 0]),
    "red":np.array([255,0,0]),
    "blue":np.array([0,0,255]),
    "yellow":np.array([255,255,0]),
    "green":np.array([7, 180, 44]),
    "purple":np.array([128,0,128]),
    "orange":np.array([255, 165, 0]),
    "pink":np.array([255, 192, 203]),
    "darkpink":np.array([173, 109, 119]),
    "lightgray":np.array([200, 200, 200]),
    "brown":np.array([123, 63, 0]),
    "black":np.array([0,0,0]),
    "white":np.array([255,255,255]),
    "gray":np.array([128,128,128]),
}

from haishoku import haillow
from haishoku import alg

class HaishokuModifed(Haishoku):
    def __init__(self):
        super().__init__()
    
    def getColorsMean(image):
        # get colors tuple with haillow module
        image_colors = get_colors(image)

        # sort the image colors tuple
        sorted_image_colors = alg.sort_by_rgb(image_colors)

        # group the colors by the accuaracy
        grouped_image_colors = alg.group_by_accuracy(sorted_image_colors)

        # get the weighted mean of all colors
        colors_mean = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    grouped_image_color = grouped_image_colors[i][j][k]
                    if 0 != len(grouped_image_color):
                        color_mean = alg.get_weighted_mean(grouped_image_color)
                        colors_mean.append(color_mean)

        # return the most 8 colors
        temp_sorted_colors_mean = sorted(colors_mean)
        if 8 < len(temp_sorted_colors_mean):
            colors_mean = temp_sorted_colors_mean[len(temp_sorted_colors_mean)-8 : len(temp_sorted_colors_mean)]
        else:
            colors_mean = temp_sorted_colors_mean

        # sort the colors_mean
        colors_mean = sorted(colors_mean, reverse=True)

        return colors_mean
    
    def showPalette(image_path):
        # get the palette first
        palette = HaishokuModifed.getPalette(image_path)

        # getnerate colors boxes
        images = []
        for color_mean in palette:
            w = color_mean[0] * 400
            color_box = haillow.new_image('RGB', (int(w), 20), color_mean[1])
            images.append(color_box)

        # generate and show the palette
        haillow.joint_image(images)

    def showDominant(image_path):
        # get the dominant color
        dominant = HaishokuModifed.getDominant(image_path)

        # generate colors boxes
        images = []
        dominant_box = haillow.new_image('RGB', (50, 20), dominant)
        for i in range(8):
            images.append(dominant_box)

        # show dominant color
        haillow.joint_image(images)


    def getDominant(image_path=None):
        # get the colors_mean
        colors_mean = HaishokuModifed.getColorsMean(image_path)
        colors_mean = sorted(colors_mean, reverse=True)

        # get the dominant color
        dominant_tuple = colors_mean[0]
        dominant = dominant_tuple[1]
        return dominant

    def getPalette(image_path=None):
        # get the colors_mean
        colors_mean = HaishokuModifed.getColorsMean(image_path)

        # get the palette
        palette_tmp = []
        count_sum = 0
        for c_m in colors_mean:
            count_sum += c_m[0]
            palette_tmp.append(c_m)

        # calulate the percentage
        palette = []
        for p in palette_tmp:
            pp = '%.2f' % (p[0] / count_sum)
            tp = (float(pp), p[1])
            palette.append(tp)

        return palette

def get_image(image):
    return image.convert("RGBA")

def get_thumbnail(image):
    image.thumbnail((256, 256))
    return image

def get_colors(image):
    """ image instance
    """
    image = get_image(image)

    """ image thumbnail
        size: 256 * 256
        reduce the calculate time 
    """
    thumbnail = get_thumbnail(image)


    """ calculate the max colors the image cound have
        if the color is different in every pixel, the color counts may be the max.
        so : 
        max_colors = image.height * image.width
    """
    image_height = thumbnail.height
    image_width = thumbnail.width
    max_colors = image_height * image_width

    image_colors = image.getcolors(max_colors)
    image_colors = list(filter(lambda x: x[1][3] > TRANSPARENCY_LIMIT, image_colors))
    image_colors = [(col[0], col[1][:3]) for col in image_colors]
    return image_colors


def get_cloth_color(image, bbox):
    """
    Function that returns the name of the color of a given item of clothing.
    Params:
    image -- original image from which to extract the item of clothing
    bbox -- bounding box including the item of clothing, in a form (x1, y1, x2, y2)
    Returns:
    color -- name of the color of the item of clothing (chosen from a list, see COLOR_DICT)
    """
    image_cropped = get_cloth_from_image(image, bbox) 
    
    img_no_background = remove(image_cropped)
    mask = img_no_background.split()[3]
    if np.sum(np.array(mask)<TRANSPARENCY_LIMIT)/np.prod(np.array(mask).shape) > BACKGROUND_PERCENTAGE_THRESHOLD: # check if background takes more than 80% of image
        final_img = image_cropped # if so, don't remove background
    else:
        final_img = img_no_background
    dominating_color = HaishokuModifed.getDominant(final_img) # get dominating color RGB
    dominating_color = np.array(dominating_color)
    distances = get_distances_from_dominating_color(dominating_color)
    color = list(COLOR_DICT.keys())[distances.index(min(distances))] # get color name that is the closest to the dominating color
    return color


def get_distances_from_dominating_color(dominating_color):
     # calculate distances to some basic colors
    distances = []
    for col_rgb in COLOR_DICT.values(): 
        distances.append(np.linalg.norm(dominating_color-col_rgb))
    return distances
    

def get_cloth_from_image(image, bbox):
     # crop image to the bounding box
    x1, y1, x2, y2 = bbox
    image_cropped = image.crop((x1, y1, x2, y2))
    return image_cropped


# load pretrained or custom model
PERSON_ID = 0
human_detector = yolov7.load('kadirnar/yolov7-v0.1', hf_model=True)

# set model parameters
human_detector.conf = 0.8  # NMS confidence threshold
human_detector.iou = 0.45  # NMS IoU threshold
human_detector.classes = PERSON_ID  # filter by class


def process_image(image):
    per_human_detections = per_human_predictor(human_detector, model, feature_extractor, image, threshold=0.8)

    for person in per_human_detections:
        for element in person["elements"]:
            element["relative_bbox"] = element["relative_bbox"].tolist()
            element["color"] = get_cloth_color(image, element["absolute_bbox"])
    
    return per_human_detections



def process_directory(path):
    image_filenames = os.listdir(path)
    directory_results = {}
    for filename in image_filenames:
        print(f"Processing {filename}", file=sys.stderr)
        try:
            filename_path = os.path.join(path, filename)
            image = Image.open(open(filename_path, "rb"))
            image = fix_channels(ToTensor()(image))
            directory_results[filename] = process_image(image)
        except Exception:
            print(f"Error processing {filename}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    return directory_results

import argparse
parser = argparse.ArgumentParser(prog="Clothing analyser", description="Analyse pictures in a directory, detecting persons, their clothes and colors of their clothes")
parser.add_argument("input_dir", help="Path to directory containing the images to analyse.")
parser.add_argument("-o", "--output-file", help="File to dump the results of the analysis. Print out to stout if not provided.")
parser.add_argument("-m", "--clothing-detection-model", default="itesl/yolos-tiny-fashionpedia-remapped", help="Huggingface model to use for clothing detection")

args = parser.parse_args()

# usage
MODEL_NAME = args.clothing_detection_model
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small') #
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

results = process_directory(args.input_dir)

if not args.output_file:
    print(json.dumps(results, indent=2))
else:
    with open(args.output_file, "w") as f:
        json.dump(results, f)
