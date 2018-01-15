import os
import requests
from io import BytesIO
from io import TextIOWrapper
import zipfile

url = "http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
UCF_train_test_split_zip = zipfile.ZipFile(
    file=BytesIO(requests.get(url).content)
)

train_list_paths = ['ucfTrainTestlist/trainlist01.txt',
                    'ucfTrainTestlist/trainlist02.txt',
                    'ucfTrainTestlist/trainlist03.txt', ]
test_list_paths = ['ucfTrainTestlist/testlist01.txt',
                   'ucfTrainTestlist/testlist02.txt',
                   'ucfTrainTestlist/testlist03.txt', ]

# 类别名称与标注序号对应表
catogory_index = {}
with BytesIO(UCF_train_test_split_zip.read("ucfTrainTestlist/classInd.txt")) as input_file:
    text = TextIOWrapper(input_file, newline="\r\n")
    for line in text.readlines():
        index, catogory = line.replace("\r\n", "").split(" ")
        catogory_index[catogory] = int(index)

# 写的很乱，总的来说就是将路径中的.avi去掉，并且将标注序号减1

os.makedirs("../../dataset_lists",exist_ok=True)
for path in train_list_paths:
    with BytesIO(UCF_train_test_split_zip.read(f"{path}")) as input_file:
        with open(f"../../dataset_lists/{os.path.basename(path)}", "w") as output_file:
            text = TextIOWrapper(input_file, newline="\r\n")
            for line in text.readlines():
                line_ = line.replace("\n", "").replace(".avi", "")
                catogory, index = line_.split(" ")
                index = int(index) - 1
                output_file.write(catogory + " " + str(index) + "\n")

for path in test_list_paths:
    with BytesIO(UCF_train_test_split_zip.read(f"{path}")) as input_file:
        with open(f"../../dataset_lists/{os.path.basename(path)}", "w") as output_file:
            text = TextIOWrapper(input_file, newline="\r\n")
            for line in text.readlines():
                output_file.write(line.replace("\r\n", "").replace(".avi", "") + " " + str(
                    catogory_index[line.split("/")[0]] - 1) + "\n")
