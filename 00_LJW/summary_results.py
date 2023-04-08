import openpyxl
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def get_points_name():
    points_name = []
    for i in '12456':
        for row in open(r'E:\LJW\mmpose\00_LJW\dataset/{}/keypoints_info/keypoint_names_{}.txt'.format(i, i), 'r',
                        encoding='utf-8').readlines():
            points_name.append(row.strip())
    return points_name


def get_skeletons():
    skeletons = []
    for i in '12456':
        for row in open(r'E:\LJW\mmpose\00_LJW\dataset/{}/keypoints_info/skeletons_{}.txt'.format(i, i), 'r',
                        encoding='utf-8').readlines():
            skeletons.append(row.strip().split())
    return skeletons


def get_OKS(pr, gt, scale, sigma):
    pr = np.reshape(pr, (-1, 3))
    gt = np.reshape(gt, (-1, 3))

    dd = (pr[:, 0] - gt[:, 0]) ** 2 + (pr[:, 1] - gt[:, 1]) ** 2
    dd = dd[gt[:, 2] > 0]

    oks = np.average(np.exp(- dd / (2 * scale * scale * sigma * sigma)))
    return oks


def get_mAP(OKSs):
    APs = []
    for i in range(10, 20):
        t = round(i * 0.05, 2)
        AP = len(OKSs[OKSs > t]) / len(OKSs)
        APs.append(AP)
    APs = np.array(APs)
    mAP = np.average(APs)
    return mAP


def get_Sigma_mAP(pr_data, gt_data, Sigmas_mAPs_save_path):
    Sigmas = []
    mAPs = []
    for j in range(10, 100):
        sigma = round(j * 0.0005, 4)
        Sigmas.append(sigma)

        OKSs = []
        for i in range(len(pr_data)):
            assert pr_data[i]['image_id'] == gt_data[i]['image_id']
            pr = np.array(pr_data[i]['keypoints'])
            gt = np.array(gt_data[i]['keypoints'])
            scale = max(gt_data[i]['bbox'][2], gt_data[i]['bbox'][3])
            assert gt.shape == pr.shape
            OKS = get_OKS(pr, gt, scale, sigma)
            OKSs.append(OKS)
        OKSs = np.array(OKSs)
        mAP = get_mAP(OKSs)
        mAPs.append(mAP)
        # print('mAP:{}'.format(round(mAP, 2)))
    book = openpyxl.Workbook()
    sh = book.active
    sh.title = 'SHeet1'
    sh['A1'] = 'Sigma'
    sh['B1'] = 'mAP'
    for i in range(len(Sigmas)):
        sh['A{}'.format(i + 2)] = Sigmas[i]
        sh['B{}'.format(i + 2)] = mAPs[i]
    save_path = Sigmas_mAPs_save_path.replace('svg', 'xlsx')
    book.save(save_path)
    book.close()
    # exit()
    Sigmas = np.array(Sigmas)
    mAPs = np.array(mAPs)

    plt.figure(figsize=(10, 7.5))
    plt.plot(Sigmas, mAPs)
    plt.savefig(Sigmas_mAPs_save_path.replace('svg', 'png'))
    plt.savefig(Sigmas_mAPs_save_path, dpi=300, format='svg')
    return Sigmas, mAPs


def main():
    src_dir = r"results"

    save_book_name = "results.xlsx"

    book = openpyxl.Workbook()
    sh = book.active
    sh.title = 'results'

    

    titles = ["backbone_name", "img_size", "batch_size", "sigma", "AP", "AR"]
    for i in range(1, 10):
        titles.append(i * 0.005)
    for i, title in enumerate(titles):
        sh[f"{chr(65 + i)}1"] = title

    for j, src_dir_name in enumerate(os.listdir(src_dir)):
        if "hrnet_w48" in src_dir_name:
            backbone_name = "hrnet"
        elif "res50" in src_dir_name:
            backbone_name = "res50"
        elif "swin_b_p4_w12" in src_dir_name:
            backbone_name = "swin"
        else:
            continue

        img_size = src_dir_name.split('_')[-2]
        batch_size = src_dir_name.split('batch')[-1]
        sigma = 2

        f = open(os.path.join(src_dir, src_dir_name, "result.txt"), "r", encoding='utf-8').readlines()
        AP = float(f[-11].strip().split()[1])
        AR = float(f[-6].strip().split()[1])
        record_list = [backbone_name, int(img_size), int(batch_size), sigma, AP, AR]

        pr_data_path = os.path.join(src_dir, src_dir_name, r"result_keypoints.json")
        gt_data_path = os.path.join(
            r"E:\LJW\mmpose\00_LJW\resized_dataset\{}\annotations\0_keypoints_test.json".format(img_size))
        Sigmas_mAPs_save_path = os.path.join(src_dir, src_dir_name, "Sigmas_mAPs.svg")
        pr_data = json.load(open(pr_data_path, 'r', encoding='utf-8'), strict=False)
        gt_data = json.load(open(gt_data_path, 'r', encoding='utf-8'), strict=False)['annotations']
        Sigmas, mAPs = get_Sigma_mAP(pr_data, gt_data, Sigmas_mAPs_save_path)

        for i in range(Sigmas.shape[0]):
            if i % 10 == 0:
                record_list.append(mAPs[i])

        for i, item in enumerate(record_list):
            sh[f"{chr(65 + i)}{j + 2}"] = item

    book.save(save_book_name)


if __name__ == '__main__':
    main()
