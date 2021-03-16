import os
txt_path_root = '/home/cxj/Desktop/data/electronic_datasets/0201数据标注/点'
save_path_root = "/home/cxj/Desktop/data/electronic_datasets/0201数据标注/点_512"

orig_lsit = [x for x in os.listdir(txt_path_root) if x.endswith(".txt")]

for txt_file in orig_lsit:
    txt_path = os.path.join(txt_path_root, txt_file)
    save_txt_path = os.path.join(save_path_root, txt_file)
    fx = open(save_txt_path, "w")
    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                # 第一行
                num_point = line
                fx.writelines(num_point)
            else:
                x1, y1 = [(t.strip()) for t in line.split()]
                # range from 0 to 1
                # if float(x1) >= 1.0 or float(y1) >= 1.0:
                #     continue
                x1, y1 = float(x1) * 512, float(y1) * 512
                cx, cy = int(x1), int(y1)
                line = str(cx) + " " + str(cy) + "\n"
                fx.writelines(line)

    fx.close()