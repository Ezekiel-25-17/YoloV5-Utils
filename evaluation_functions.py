from decimal import DivisionByZero
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def list_to_coord(x):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    
    x = [[x0, x2], [x1, x2], [x1, x3], [x0, x3]]
    return x


def calculate_iou(box_1, box_2): # box = [x_min, x_max, y_min, y_max]
    box_1 = list_to_coord(box_1)
    box_2 = list_to_coord(box_2)
    
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def convert(x, label=True):
    x_min = int((float(x[1])-(float(x[3])/2))*640)
    x_max = int((float(x[1])+(float(x[3])/2))*640)

    y_min = int((float(x[2])-(float(x[4])/2))*640)
    y_max = int((float(x[2])+(float(x[4])/2))*640)
    
    if not label:
        return x_min, x_max, y_min, y_max
    else:
        return x_min, x_max, y_min, y_max, int(x[0])
    

def calculate_area(prediction):
    return (prediction[1] - prediction[0] + 1) * (prediction[3] - prediction[2] + 1)


def compute_overall_metrics(true_path, pred_path, yolo, confidence, iou_threshold=0.45):

    try:
        true = pd.read_csv(true_path).drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    except KeyError:
        try:
            true = pd.read_csv(true_path).drop('Unnamed: 0', axis=1)
        except KeyError:
            true = pd.read_csv(true_path)
    try:
        pred = pd.read_csv(pred_path).drop('Unnamed: 0', axis=1)
    except KeyError:
        pred = pd.read_csv(pred_path)

    tp = 0
    fp = 0
    fn = 0
    ious = []

    imgs = set(list(true['photo_filename']))

    for filename in imgs:
        tmp_true = true[true['photo_filename'] == filename].reset_index(drop=True)
        try:
            tmp_pred = pred[pred['photo_filename'] == filename].reset_index(drop=True)
        except KeyError:
            fn += 1
            continue

        for i, row in tmp_true.iterrows():
            true_box = [row['xmin'], row['xmax'], row['ymin'], row['ymax']]
            match = False
            while not match:
                for i, row in tmp_pred.iterrows():
                    pred_box = [row['xmin'], row['xmax'], row['ymin'], row['ymax']]
                    iou = calculate_iou(true_box, pred_box)
                    if iou >= iou_threshold:
                        match = True
                        tp += 1
                        ious.append(iou)
                        tmp_pred.drop(i, axis=0)
                        break
                    else:
                        continue
                if not match:
                    fn += 1
                    match = True
        fp += len(tmp_pred)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    if len(ious) > 0:
        average_iou = np.mean(ious)
    else:
        average_iou = np.nan

    print(f'Yolo: {yolo}')
    print(f'Confidence: {confidence}')
    print(f'Number of Images: {len(true.groupby("photo_filename").count())}')
    print(f'Number of (True) Logos {len(true)}')
    print(f'Number of Logos Detected: {len(pred)}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'Average IoU: {average_iou}')


def convert_metrics(txt_path, out_path, n_runs):
    with open(txt_path) as f:
        lines = f.readlines()

    df = pd.DataFrame(columns=['yolo', 'confidence', 'n_logos', 'n_logos_detected', 'precision', 'recall', 'f1', 'average_iou'])

    for i in range(0, n_runs*10, 10):
        d = {'yolo' : lines[0+i].split()[1],
            'confidence' : lines[1+i].split()[1],
            'n_logos' : int(lines[3+i].split()[4]),
            'n_logos_detected' : int(lines[4+i].split()[4]),
            'precision' : np.float(lines[5+i].split()[1]),
            'recall' : np.float(lines[6+i].split()[1]),
            'f1' : np.float(lines[7+i].split()[1]),
            'average_iou' : np.float(lines[8+i].split()[2])}

        df = df.append(d, ignore_index=True)
    df = df.sort_values('f1', ascending=False)
    df.to_csv(out_path, index=False)
        

def compute_unseen_metrics(true_path, pred_path, out_path, iou_threshold=0.45):

    try:
        true = pd.read_csv(true_path).drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    except KeyError:
        try:
            true = pd.read_csv(true_path).drop('Unnamed: 0', axis=1)
        except KeyError:
            true = pd.read_csv(true_path)
    try:
        pred = pd.read_csv(pred_path).drop('Unnamed: 0', axis=1)
    except KeyError:
        pred = pd.read_csv(pred_path)

    d = {}

    imgs = set(list(true['photo_filename']))

    for filename in imgs:
        tmp_true = true[true['photo_filename'] == filename].reset_index(drop=True)
        try:
            tmp_pred = pred[pred['photo_filename'] == filename].reset_index(drop=True)
        except KeyError:
            for i, row in tmp_true.iterrows():
                logo = row['class']
                if logo not in d.keys():
                    d.update({logo : {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}})
                d[logo]['fn'] += 1
            continue

        for i, row in tmp_true.iterrows():

            logo = row['class']
            if logo not in d.keys():
                d.update({logo : {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}})
            true_box = [row['xmin'], row['xmax'], row['ymin'], row['ymax']]
            match = False
            while not match:
                for i, row in tmp_pred.iterrows():
                    pred_box = [row['xmin'], row['xmax'], row['ymin'], row['ymax']]
                    iou = calculate_iou(true_box, pred_box)
                    if iou >= iou_threshold:
                        match = True
                        d[logo]['tp'] += 1
                        d[logo]['ious'].append(iou)
                        tmp_pred.drop(i, axis=0)
                        break
                    else:
                        continue
                if not match:
                    d[logo]['fn'] += 1
                    match = True
        d[logo]['fp'] += len(tmp_pred)

    df = pd.DataFrame(columns=['logo', 'precision', 'recall', 'f1', 'average_iou', 'n_logos', 'tp', 'fp', 'fn'])

    for key in d.keys():
        try:
            precision = d[key]['tp'] / (d[key]['tp'] + d[key]['fp'])
        except (ZeroDivisionError, DivisionByZero):
            precision = 0
        try:
            recall = d[key]['tp'] / (d[key]['tp'] + d[key]['fn'])
        except (ZeroDivisionError, DivisionByZero):
            recall = 0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except (ZeroDivisionError, DivisionByZero):
            f1 = 0
        if len(d[key]['ious']) > 0:
            average_iou = np.mean(d[key]['ious'])
        else:
            average_iou = np.nan

        d_tmp = {'logo' : key,
                'precision' : precision,
                'recall' : recall,
                'f1' : f1,
                'average_iou' : average_iou,
                'n_logos' : d[key]['tp'] + d[key]['fp'] + d[key]['fn'],
                'tp' : d[key]['tp'],
                'fp' : d[key]['fp'],
                'fn' : d[key]['fn']}

        df = df.append(d_tmp, ignore_index=True)

    df.to_csv(out_path, index=False)


def compute_brand_metrics(true_path, pred_path, out_path, brands_dict, iou_threshold=0.45, noise=False):

    try:
        true = pd.read_csv(true_path).drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    except KeyError:
        try:
            true = pd.read_csv(true_path).drop('Unnamed: 0', axis=1)
        except KeyError:
            true = pd.read_csv(true_path)
    try:
        pred = pd.read_csv(pred_path).drop('Unnamed: 0', axis=1)
    except KeyError:
        pred = pd.read_csv(pred_path)

    d = {}

    imgs = set(list(true['photo_filename']))

    for filename in imgs:
        tmp_true = true[true['photo_filename'] == filename].reset_index(drop=True)
        try:
            tmp_pred = pred[pred['photo_filename'] == filename].reset_index(drop=True)
        except KeyError:
            for i, row in tmp_true.iterrows():
                logo = row['class']
                if logo not in d.keys():
                    d.update({logo : {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}})
                d[logo]['fn'] += 1
            continue

        for i, row in tmp_true.iterrows():
            logo = row['class']
            if logo not in d.keys():
                d.update({logo : {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}})
            true_box = [row['xmin'], row['xmax'], row['ymin'], row['ymax']]
            match = False
            while not match:
                for i, row in tmp_pred.iterrows():
                    pred_logo = brands_dict[row['class']]
                    pred_box = [row['xmin'], row['xmax'], row['ymin'], row['ymax']]
                    iou = calculate_iou(true_box, pred_box)
                    if (iou >= iou_threshold) and (logo == pred_logo):
                        match = True
                        d[logo]['tp'] += 1
                        d[logo]['ious'].append(iou)
                        tmp_pred.drop(i, axis=0)
                        break
                    else:
                        continue
                if not match:
                    d[logo]['fn'] += 1
                    match = True
        d[logo]['fp'] += len(tmp_pred)

    df = pd.DataFrame(columns=['logo', 'precision', 'recall', 'f1', 'average_iou', 'n_logos', 'tp', 'fp', 'fn'])

    for key in d.keys():
        try:
            precision = d[key]['tp'] / (d[key]['tp'] + d[key]['fp'])
        except (ZeroDivisionError, DivisionByZero):
            precision = 0
        try:
            recall = d[key]['tp'] / (d[key]['tp'] + d[key]['fn'])
        except (ZeroDivisionError, DivisionByZero):
            recall = 0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except (ZeroDivisionError, DivisionByZero):
            f1 = 0
        if len(d[key]['ious']) > 0:
            average_iou = np.mean(d[key]['ious'])
        else:
            average_iou = np.nan

        d_tmp = {'logo' : key,
                'precision' : precision,
                'recall' : recall,
                'f1' : f1,
                'average_iou' : average_iou,
                'n_logos' : d[key]['tp'] + d[key]['fp'] + d[key]['fn'],
                'tp' : d[key]['tp'],
                'fp' : d[key]['fp'],
                'fn' : d[key]['fn']}

        df = df.append(d_tmp, ignore_index=True)

    df.to_csv(out_path, index=False)
        