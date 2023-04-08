import os



def main():

    src_dir = r"E:/LJW/mmpose/00_LJW/000_new_results_only_one"
    dst_dir = r"E:/LJW/Git/my_mmpose/mmpose/00_LJW/results"


    for src_dir_name  in os.listdir(src_dir):

        src_dir_1 = os.path.join(src_dir, src_dir_name)

        dst_dir_1 = os.path.join(dst_dir, src_dir_name)
        out_path = os.path.join(dst_dir_1, r"result.json")

        out_txt_path = os.path.join(dst_dir_1, r"result.txt")
        if os.path.exists(out_txt_path):
            continue

        if not os.path.exists(dst_dir_1):
            os.makedirs(dst_dir_1)


        if "hrnet_w48" in src_dir_1:
            model_path  = os.path.join(src_dir_1, os.listdir(src_dir_1)[0])
            src_file_path = r"E:/LJW/Git/my_mmpose/mmpose/configs/_hrnet_w48.py"
        elif "res50" in src_dir_1:
            model_path  = os.path.join(src_dir_1, os.listdir(src_dir_1)[0])
            src_file_path = r"E:/LJW/Git/my_mmpose/mmpose/configs/_res50.py"
        elif "swin_b_p4_w12" in src_dir_1:
            model_path  = os.path.join(src_dir_1, os.listdir(src_dir_1)[0])
            src_file_path = r"E:/LJW/Git/my_mmpose/mmpose/configs/_swin_b_p4_w12.py"
        else:
            continue

        img_size = src_dir_name.split('_')[-2]
        batch_size = 1
        sigma = 2
        worker = 1
        interval= 802 // 1 // batch_size // 50 * 50

        temp_config_path = r"E:/LJW/Git/my_mmpose/mmpose/00_LJW/temp.py"

        f = open(src_file_path, "r", encoding="utf-8").read()

        f = f.replace("img_size = 1024", f"img_size = {img_size}")
        f = f.replace("batch_size = 2", f"batch_size = {batch_size}")
        f = f.replace("workers = 2", f"workers = {worker}")
        f = f.replace("sigma = 2", f"sigma = {sigma}")
        f = f.replace("interval= 802 // batch_size // 50 * 50", f"interval= {interval}")
        f = f.replace("'_base_", f"'E:/LJW/Git/my_mmpose/mmpose/configs/_base_")
        f = f.replace("/kaggle/input/tower-dataset-2", f"E:/LJW/mmpose/00_LJW")

        f2 = open(temp_config_path, "w", encoding="utf-8")
        f2.write(f)
        f2.close()


        s = f"python E:/LJW/Git/my_mmpose/mmpose/tools/test.py {temp_config_path} {model_path} --out {out_path} --work-dir {dst_dir_1}"
        # print(s)
        os.system("conda activate openmmlab")

        # os.system(s)

        f = os.popen(s, "r")
        result = f.read()

        f2 = open(out_txt_path, "w", encoding='utf-8')
        f2.write(result)
        f2.close()

        # print(result)
        # exit()



if __name__ == '__main__':
    pass