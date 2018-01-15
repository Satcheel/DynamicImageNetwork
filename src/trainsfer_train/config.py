class Dataset:
    split_order = 1

    class MaxImage:
        mean = [0.5968608856201172, 0.5741104483604431, 0.5484067797660828]
        std = [0.2581653892993927, 0.25172218680381775, 0.251637727022171]

        class Train:
            directory = "~/datasets/UCF_extract/MaxImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/trainlist01.txt"

        class Validation:
            directory = "~/datasets/UCF_extract/MaxImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/testlist01.txt"

    class MeanImage:
        mean = [0.3952347934246063, 0.3790866732597351, 0.34863102436065674]
        std = [0.2083127647638321, 0.20260944962501526, 0.19819821417331696]

        class Train:
            directory = "~/datasets/UCF_extract/MeanImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/trainlist01.txt"

        class Validation:
            directory = "~/datasets/UCF_extract/MeanImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/testlist01.txt"

    class StaticImage:
        mean = [0.39487960934638977, 0.37888103723526, 0.348604291677475]
        std = [0.24151542782783508, 0.23447425663471222, 0.22991050779819489]

        class Train:
            directory = "~/datasets/UCF_extract/StaticImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/trainlist01.txt"

        class Validation:
            directory = "~/datasets/UCF_extract/StaticImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/testlist01.txt"

    class SDIImage:
        mean = [0.5104864239692688, 0.5113173127174377, 0.5112648010253906]
        std = [0.09384600818157196, 0.0896846279501915, 0.08940195292234421]

        class Train:
            directory = "~/datasets/UCF_extract/SDIImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/trainlist01.txt"

        class Validation:
            directory = "~/datasets/UCF_extract/SDIImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/testlist01.txt"

    class MDIImage:
        mean = [0.5098264813423157, 0.5102994441986084, 0.509983479976654]
        std = [0.08257845789194107, 0.07896881550550461, 0.07860155403614044]

        class Train:
            directory = "~/datasets/UCF_extract/MDIImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/trainlist01.txt"

        class Validation:
            directory = "~/datasets/UCF_extract/MDIImage"
            label = "/home/chenyaofo/workspace/DynamicImageNetwork/dataset_lists/testlist01.txt"


class Strategy:
    class Train:
        batch_size = 64
        learning_rate = 0.01
        max_epoches = 50

    class Validation:
        batch_size = 64
