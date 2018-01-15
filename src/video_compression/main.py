import os
from math import ceil
import multiprocessing
from video_compression.video import Video,VideoReadError
from video_compression.core import *


def chunks(arr, m):
    n = int(ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


datasets_dir = "/home/chenyaofo/dataset_lists/UCF101"
project_dir = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists"

process_dict = {
    "mean_image": (Mean, MeanImage),
    "max_image": (Max, MaxImage),
    "static_image": (Static, StaticImage),
    "dynamic_image_SDI": (SDI, DynamicImage),
    "dynamic_image_MDI": (MDI, DynamicImage),
}


def informative_image_wrap(paths):
    for catagory_name, path in paths:
        try:
            video = Video(path)
        except VideoReadError:
            continue
        pure_filename, _ = os.path.splitext(os.path.basename(path))
        for tag,(selector,extractor) in process_dict.items():
            print(f"processing {pure_filename} to {tag}")
            os.makedirs(os.path.join(project_dir, f"{tag}", catagory_name), exist_ok=True)
            for i,_slice in enumerate(selector(video.n_frames)):
                extractor(video.get_frames(_slice)).process().save(os.path.join(project_dir, f"{tag}", catagory_name, pure_filename + f"_{i:02d}.jpg"))

if __name__ == '__main__':
    paths = []
    for catagory_name in os.listdir(datasets_dir):
        for video_name in os.listdir(os.path.join(datasets_dir, catagory_name)):
            paths.append((catagory_name, os.path.join(datasets_dir, catagory_name, video_name)))
    paths_split = chunks(paths, multiprocessing.cpu_count())
    # informative_image_wrap(paths)
    for i in range(multiprocessing.cpu_count()):
        multiprocessing.Process(target=informative_image_wrap, args=(paths_split[i],)).start()
