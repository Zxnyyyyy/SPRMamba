from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt
import os

def evaluate_phase(all_prediction, all_labels):
    recall_phase = metrics.recall_score(all_prediction, all_labels, average='macro')
    precision_phase = metrics.precision_score(all_prediction, all_labels, average='macro')
    jaccard_phase = metrics.jaccard_score(all_prediction, all_labels, average='macro')
    return recall_phase, precision_phase, jaccard_phase

def evaluate_video(prediction, labels):
    recall_video = metrics.recall_score(prediction, labels, average='macro')
    precision_video = metrics.precision_score(prediction, labels, average='macro')
    jaccard_video = metrics.jaccard_score(prediction, labels, average='macro')
    return recall_video, precision_video, jaccard_video

def save_prediction(save_path, prediction, fps, phase_dict_key):
    f = open(save_path, 'w')
    cnt = 0
    f.write('Frame\tPhase\n')
    for j in range(len(prediction)):
        p_cpu = prediction.cpu()
        p_num = p_cpu.numpy()
        for ind in range(cnt * fps, (cnt + 1) * fps):
            f.write(str(ind) + '\t')
            f.write(str(phase_dict_key[p_num[cnt]]) + '\t')
            f.write('\n')
        cnt = cnt + 1
    f.close()

def segment_bars(save_path, label):
    rows, cols = label.shape
    color_map = plt.cm.tab10
    fig = plt.figure(figsize=(20, 2))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=10)
    plt.subplot()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(label, **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def save_png_single(save_path, prediction):
    prediction = np.array(prediction).reshape(1, -1)
    segment_bars(save_path=save_path, label=prediction)

def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths

def save_png_all(phase_dir, pic_save_dir, phase_dict, fps=50):
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    phase_file_names, phase_file_paths = get_files(phase_dir)

    labels = []
    num_each = []

    for j in range(len(phase_file_names)):
        phase_file = open(phase_file_paths[j])
        content = phase_file.read()
        last = content.split()[-2]
        phase_file = open(phase_file_paths[j])

        info_all = []
        first_line = True
        for phase_line in phase_file:
            phase_split = phase_line.split()
            if first_line:
                first_line = False
                continue
            if int(phase_split[0]) % fps == 0 and int(phase_split[0]) != int(last):
                info_each = []
                labels.append(phase_dict[phase_split[1]])
                info_each.append(phase_dict[phase_split[1]])
                info_all.append(info_each)
        num_each.append(len(info_all))
        start_vidx = []
        count = 0
        for i in range(len(num_each)):
            start_vidx.append(count)
            count += num_each[i]
        we_use_start_idx = [x for x in range(len(num_each))]

        for i in we_use_start_idx:
            labels_phase = []
            for j in range(start_vidx[i], start_vidx[i] + num_each[i]):
                labels_phase.append(labels[j])
            label = np.array(labels_phase).reshape(1, -1)
            segment_bars(save_path=os.path.join(pic_save_dir, str(
                os.path.splitext(os.path.basename(phase_file_paths[i]))[0]) + '.png'), label=label)