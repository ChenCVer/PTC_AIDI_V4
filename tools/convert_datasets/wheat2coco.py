import pandas as pd
import json
import warnings

# 数据说明参考: https://www.pianshen.com/article/663915805/

warnings.filterwarnings("ignore")

marking = pd.read_csv(r'/home/cxj/Desktop/data/global-wheat-detection/train.csv')
marking.head(2)

wheat_data = dict()

wheat_data["info"] = {'description': 'wheat 2020 Dataset',
                      'url': 'https://www.kaggle.com/c/global-wheat-detection',
                      'version': '1.0',
                      'year': 2020,
                      'contributor': 'COCO Consortium',
                      'date_created': '2020/11/27'}

wheat_data["license"] = "PTC_license"
wheat_data["images"] = []
wheat_data["annotations"] = []
wheat_data["categories"] = [{"name": "wheat", "id": 1}]

img_list = []

for indexs in marking.index:
    attr_list = marking.loc[indexs].values
    img_id = attr_list[0]
    width = attr_list[1]
    height = attr_list[2]
    bbox = eval(attr_list[3])

    if img_id not in img_list:
        image_dict = {
            "file_name": img_id + ".jpg",
            "width": int(width),
            "height": int(height),
            "id": img_id
        }
        wheat_data["images"].append(image_dict)
        img_list.append(img_id)

    annotation_dict = {
        "segmentation": [[]],
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "image_id": img_id,
        "bbox": [bbox[0], bbox[1],
                 bbox[2], bbox[3]],
        "category_id": 1,
        "id": indexs
    }
    print("bbox = ", annotation_dict["bbox"])
    wheat_data["annotations"].append(annotation_dict)


with open("wheat_train.json", "w") as f:
    f.write(json.dumps(wheat_data, ensure_ascii=False))