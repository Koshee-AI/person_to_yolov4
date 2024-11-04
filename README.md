# person_to_yolov4

This script uses **Tensorflow** and **SSD Mobilenet V2 (COCO)** to recognize
people and then update a yolov4 dataset with captions and bboxes that are
appropriate to be loaded into DarkMark.

## How to use
1. Download the repository
```
cd ~/koshee
git clone git@github.com:Koshee-AI/person_to_yolov4.git
```

2. Download the protect-scripts repository (required to link to captions list)

```
cd ~/koshee
git clone git@github.com:Koshee-AI/protect-scripts.git
```

2. Install all requirements

```
cd ~/koshee/person_to_yolov4
./run.sh -h
```

4. Run person_to_yolov4.py

**Get Help**
```
python ~/koshee/person_to_yolov4/person_to_yolov4.py -h
```

**Typical Usage**
```
cd new_directory_with_images_dir
python ~/koshee/person_to_yolov4/person_to_yolov4.py -g 20 "a person moving an item under their shirt"
```

The above would label images inside of `new_directory_with_images_dir/images` with a 20% bigger box
than default person selection and label that box with the caption `a person moving an item under their shirt`