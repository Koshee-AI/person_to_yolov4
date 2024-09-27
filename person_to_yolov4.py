import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import requests
import json
import argparse
import time

user_home = os.path.expanduser("~")
model_path = f"{user_home}/koshee/.checkpoints/sd_mobilenetv2_coco"

sys.path.append(f"{user_home}/koshee/protect-scripts")

import utils.caption_lib


def parse_args(args):
  parser = argparse.ArgumentParser(
    description="Generate json/txt files for supporting yolov4 datasets",
    add_help=True
  )
  parser.add_argument('-d','-p', '--path', action='store', default='.', dest='path', help='the directory to update')
  parser.add_argument('--fc','--force-category', action='store', dest='force_category', default='small', help='force an anomaly category, despite filename')
  parser.add_argument('-g','--growth','--percent', action='store', type=int, dest='growth', default=10, help='grow the person selector by a percent')
  parser.add_argument('-s','--scale', action='store', type=float, dest='scale', default=1.0, help='scale')
  parser.add_argument('new_captions', nargs=argparse.REMAINDER, help='captions to set images in the range to')
  return parser.parse_args(args)
  

# Download model if not exist
def download_model():
  url = "https://drive.google.com/uc?id=1Ml260620LIKa-OrWqdzv_z99NJKixt3W"


  os.makedirs(model_path, exist_ok=True)

  output_path = f"{model_path}/saved_model.pb"

  response = requests.get(url, stream=True)

  with open(output_path, "wb") as output_file:
    for chunk in response.iter_content(chunk_size=8192):
      if chunk:
        output_file.write(chunk)

# Function to perform object detection and crop the image
def object_detection(input_image_path, options, caption_lookup):

  # Read the input image
  image = cv2.imread(input_image_path)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_expanded = np.expand_dims(image_rgb, axis=0).astype(np.uint8)

  # Perform inference
  detections = model.signatures["serving_default"](tf.constant(image_expanded))

  # Extract coordinates of detected persons
  detection_classes = detections['detection_classes'][0].numpy()
  detection_boxes = detections['detection_boxes'][0]
  person_indices = tf.where(tf.equal(detection_classes, 1))[:, 0]
  person_coords = tf.gather(detection_boxes, person_indices).numpy()

  image_path = os.path.dirname(input_image_path)
  image_name = os.path.basename(input_image_path)
  base_name = os.path.splitext(image_name)[0]

  common_name = f"{image_path}/{base_name}"

  if len(options.new_captions) != 1:
    print(f"Requires 1 caption. Found {len(options.new_captions)}")
    exit(0)

  caption = options.new_captions[0]
  caption_idx = caption_lookup.get(caption)

  if caption_idx == None:
    print(f"Caption {caption} was not found in the category {options.force_category}")
    exit(0)

  data = {
    "completely_false" : False,
    "image" : {
      "height": image.shape[0],
      "scale": options.scale,
      "width": image.shape[1],
    },
    "mark" : [
    ],
    "timestamp": time.time(),
    "version": "1.9.4-1"
  }

  txt_file = ""

  # Crop the original image based on the detected persons' coordinates
  for i, coords in enumerate(person_coords):
    ymin, xmin, ymax, xmax = coords

    # Ensure valid coordinates
    if 0 <= ymin < ymax <= image.shape[0] and 0 <= xmin < xmax <= image.shape[1]:
      if ymin - (ymin * 0.1) >= 0:
        ymin = int((ymin - (ymin * 0.1)) * image.shape[0])
      else:
        ymin = int(ymin * image.shape[0])
      
      if int((ymax + (ymax * 0.1)) * image.shape[0]) <= image.shape[0]:
        ymax = int((ymax + (ymax * 0.1)) * image.shape[0])
      else:
        ymax = int(ymax * image.shape[0])
      
      if xmin - (xmin * 0.1) >= 0:
        xmin = int((xmin - (xmin * 0.1)) * image.shape[1])
      else:
        xmin = int(xmin * image.shape[1])
      
      if int((xmax + (xmax * 0.1)) * image.shape[1]) <= image.shape[1]:
        xmax = int((xmax + (xmax * 0.1)) * image.shape[1])
      else:
        xmax = int(xmax * image.shape[1])
      
      width = xmax - xmin
      height = ymax - ymin

      new_width = int(width * ((100 + float(options.growth)) / 100))
      new_height = int(height * ((100 + float(options.growth)) / 100))

      new_width_diff = new_width - width
      new_height_diff = new_height - height

      ymin -= int(new_height_diff * 0.5)
      xmin -= int(new_width_diff * 0.5)

      ymax += int(new_height_diff * 0.5)
      xmax += int(new_width_diff * 0.5)

      xmid = int(xmin + new_width / 2)
      ymid = int(ymin + new_height / 2)

      cropped_person = image[ymin:ymax, xmin:xmax, :]

      # Check if the cropped_person array is not empty
      if not cropped_person.size == 0:
        
        xmin_norm = float(xmin) / image.shape[1]
        xmax_norm = float(xmax) / image.shape[1]

        ymin_norm = float(ymin) / image.shape[0]
        ymax_norm = float(ymax) / image.shape[0]

        width_norm = float(new_width) / image.shape[1]
        height_norm = float(new_height) / image.shape[0]

        xmid_norm = float(xmid) / image.shape[1]
        ymid_norm = float(ymid) / image.shape[0]

        person = {
          "class_idx": caption_idx,
          "name": caption,
          "points": [
            {
              "int_x": xmin,
              "int_y": ymin,
              "x": xmin_norm,
              "y": ymin_norm,
            },
            {
              "int_x": xmax,
              "int_y": ymin,
              "x": xmax_norm,
              "y": ymin_norm,
            },
            {
              "int_x": xmax,
              "int_y": ymax,
              "x": xmax_norm,
              "y": ymax_norm,
            },
            {
              "int_x": xmin,
              "int_y": ymax,
              "x": xmin_norm,
              "y": ymax_norm,
            },
          ],
          "rect" : {
            "h": new_height / image.shape[0],
            "int_h": new_height,
            "int_w": new_width,
            "int_x": xmin,
            "int_y": ymin,
            "w": new_width / image.shape[1],
            "x": xmin_norm,
            "y": ymin_norm,
          },
        }

        data["mark"].append(person)

        txt_file += f"{caption_idx} {xmid_norm} {ymid_norm} {width_norm} {height_norm}\n"

  with open(f"{common_name}.json", "w") as f:
    json.dump(data, f, indent=2)

  with open(f"{common_name}.txt", "w") as f:
    print(txt_file, file=f)
    

# Explicitly specify the GPU device
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialize model variable
model = None

if __name__ == "__main__":
  
  options = parse_args(sys.argv[1:])
  default_captions = utils.caption_lib.get_default_captions()
  caption_list = default_captions[options.force_category]
  caption_lookup = utils.caption_lib.build_class_index_lookup(caption_list)


  if os.path.exists(f"{model_path}/saved_model.pb"):
    print("LOG: SSD Mobilenet V2 COCO model found.")
  else:
    print("LOG: SSD Mobilenet V2 COCO model not found. Downloading...")
    download_model()
    print("LOG: SSD Mobilenet V2 COCO model downloaded.")

  model = tf.saved_model.load(f"{model_path}")
  
  print("LOG: Running...")
  # Process all images in the input folder
  if os.path.isdir(options.path):

    images_dir = f"{options.path}/images"

    if os.path.isdir(images_dir):

      for filename in os.listdir(images_dir):
        print(f"  evaluating {filename}")
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".webp") or filename.endswith(".bmp"):
          input_image_path = os.path.join(images_dir, filename)
          object_detection(input_image_path, options, caption_lookup)
    else:
      print(f"  Unable to use this path ({options.path}). We need a path with an images directory.")
    print("LOG: Done.")
