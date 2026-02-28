import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels, write_domain_img_file2txt, split_domain_txt2txt
import os
import logging
from utils.core50data import CORE50
from collections import defaultdict  # 新增类别计数器 debug使用


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None



class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)


class iCore50(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    train_split_ratio = 0.7
    train_session_seq = ['s11', 's4', 's2', 's9', 's1', 's6', 's5', 's8']
    test_session_seq = ['s3', 's7', 's10']
    test_split_file_prefix = 'TEST-set'

    def __init__(self, args, split_train4test=False, split_test4test=False):
        self.args = args
        class_order = np.arange(8 * 50).tolist()
        self.class_order = class_order

        self.split_train4test = split_train4test
        self.split_test4test = split_test4test

    def download_data(self):
        train_session_id = self.args.get('task_name', list(range(8)))

        self.train_session_seq = [self.train_session_seq[i] for i in train_session_id]
        logging.info("Training sequence of domains: {}".format(self.train_session_seq))

        datagen = CORE50(root=self.args["data_path"], scenario="ni", train_seq=train_session_id)

        dataset_list = []
        img_list = []
        label_list = []

        if not self.split_train4test:
            for i, train_batch in enumerate(datagen):
                img_list_, label_list_ = train_batch  # img_list_: RGB image data
                print("Loading {}: {} ".format(self.train_session_seq[i], len(img_list_)))
                label_list_ += i * 50
                img_list_ = img_list_.astype(np.uint8)
                # dataset_list.append([imglist, labellist])
                img_list.append(img_list_)
                label_list.append(label_list_)
            train_x = np.concatenate(img_list)
            train_y = np.concatenate(label_list)
            self.train_data = train_x
            self.train_targets = train_y

            test_x, test_y = datagen.get_test_set()
            test_x = test_x.astype(np.uint8)
            self.test_data = test_x
            self.test_targets = test_y
        else:
            self.write2txt()
            train_idx_lst, test_idx_lst = self.read2array()
            train_img_lst = []
            train_label_arr = np.array([], dtype=np.int32)
            test_img_lst = []
            test_label_arr = np.array([], dtype=np.int32)

            for i, train_batch in enumerate(datagen):
                img_list_, label_list_ = train_batch  # img_list_: RGB image data
                print("Loading {}: {} ".format(self.train_session_seq[i], len(img_list_)))
                label_list_ += i * 50
                img_list_ = img_list_.astype(np.uint8)

                train_img_lst.append(img_list_[train_idx_lst[i]])
                train_label_arr = np.append(train_label_arr, label_list_[train_idx_lst[i]])
                test_img_lst.append(img_list_[test_idx_lst[i]])
                test_label_arr = np.append(test_label_arr, label_list_[test_idx_lst[i]])

            if self.split_test4test:
                img_arr, label_arr = datagen.get_test_set()
                img_arr = img_arr.astype(np.uint8)
                train_img_lst.append(img_arr[train_idx_lst[-1]])
                train_label_arr = np.append(train_label_arr, label_arr[train_idx_lst[-1]])
                test_img_lst.append(img_arr[test_idx_lst[-1]])
                test_label_arr = np.append(test_label_arr, label_arr[test_idx_lst[-1]])

            self.train_data = np.concatenate(train_img_lst)
            self.train_targets = train_label_arr

            self.test_data = np.concatenate(test_img_lst)
            self.test_targets = test_label_arr

    def write2txt(self):
        datagen = CORE50(root=self.args["data_path"], scenario="ni")

        if self.split_test4test and os.path.exists(os.path.join(self.args["data_path"],
                                                                self.test_split_file_prefix + '_test_idx.txt')):
            return

        if not os.path.exists(os.path.join(self.args["data_path"], self.train_session_seq[-1]+'_test_idx.txt')):
            print("Writing train-test split index to txt file ...")

            for idx, train_batch in enumerate(datagen):
                img_list_, label_list_ = train_batch
                label_list_ += idx * 50

                train_idx_ = np.random.choice(len(img_list_), int(len(img_list_) * self.train_split_ratio), replace=False)
                test_idx_ = np.setdiff1d(np.arange(len(img_list_)), train_idx_)
                with open(os.path.join(self.args["data_path"], self.train_session_seq[idx]+'_train_idx.txt'), 'w') as f:
                    for i in train_idx_:
                        f.write(str(i) + '\n')
                with open(os.path.join(self.args["data_path"], self.train_session_seq[idx]+'_test_idx.txt'), 'w') as f:
                    for i in test_idx_:
                        f.write(str(i) + '\n')

        if self.split_test4test:
            test_x, test_y = datagen.get_test_set()
            train_idx_ = np.random.choice(len(test_x), int(len(test_x) * self.train_split_ratio), replace=False)
            test_idx_ = np.setdiff1d(np.arange(len(test_x)), train_idx_)
            with open(os.path.join(self.args["data_path"], self.test_split_file_prefix + '_train_idx.txt'), 'w') as f:
                for i in train_idx_:
                    f.write(str(i) + '\n')
            with open(os.path.join(self.args["data_path"], self.test_split_file_prefix + '_test_idx.txt'), 'w') as f:
                for i in test_idx_:
                    f.write(str(i) + '\n')

    def read2array(self):
        def _read2array(seq, train_idx:list, test_idx:list):
            for idx in seq:
                with open(os.path.join(self.args["data_path"], idx+'_train_idx.txt'), 'r') as f:
                    train_idx.append([np.int32(i) for i in f.readlines()])
                with open(os.path.join(self.args["data_path"], idx+'_test_idx.txt'), 'r') as f:
                    test_idx.append([np.int32(i) for i in f.readlines()])
            return train_idx, test_idx

        train_idx, test_idx = [], []
        train_idx, test_idx = _read2array(self.train_session_seq, train_idx, test_idx)

        if self.split_test4test:
            train_idx, test_idx = _read2array(self.test_session_seq, train_idx, test_idx)

        return train_idx, test_idx


class iDomainNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = (np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.nb_sessions = args["total_sessions"]
        self.cl_n_inc = self.args["increment"]
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False

        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        def _read_data(image_list_paths) -> (np.ndarray, np.ndarray):
            imgs = []
            for taskid, image_list_path in enumerate(image_list_paths):
                if taskid >= self.nb_sessions:
                    break
                with open(image_list_path) as f:
                    image_list = f.readlines()
                # 重写 target class := original value + taskid * args["increment"]
                for entry in image_list:
                    img_label = int(entry.split()[1])
                    if self.class_incremental:
                        if img_label < taskid * self.cl_n_inc or img_label >= (taskid + 1) * self.cl_n_inc:
                            continue
                    elif img_label > self.cl_n_inc:
                        raise ValueError("class_incremental is False, but img_label > cl_n_inc")
                    else:  # correct the label for DIL tasks
                        img_label = img_label + taskid * self.cl_n_inc
                    imgs.append((entry.split()[0], img_label))
            # class_counter = defaultdict(int)  # 新增类别计数器 debug使用
            img_x, img_y = [], []
            for item in imgs:
                img_x.append(os.path.join(self.image_list_root, item[0]))
                img_y.append(item[1])

            return np.array(img_x), np.array(img_y)

        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        self.train_data, self.train_targets = _read_data(image_list_paths)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        self.test_data, self.test_targets = _read_data(image_list_paths)

class iMedMNISTv2(iData):
    use_path = False
    
    def __init__(self, args):
        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        self.args = args
        class_order = np.arange(sum(args.get("increment_per_session", [args["increment"]]*args["total_sessions"]))).tolist()
        self.class_order = class_order
        self.nb_sessions = args["total_sessions"]
        # 修改后：支持不同session的不同增量
        self.cl_n_inc = args.get("increment_per_session", [self.args["increment"]]*self.nb_sessions)
        # 校验增量配置长度
        if len(self.cl_n_inc) != self.nb_sessions:
            raise ValueError(f"increment_per_session长度({len(self.cl_n_inc)})与total_sessions({self.nb_sessions})不匹配")
        
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False

        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        import medmnist
        from medmnist import INFO, Evaluator
        train_datas = []
        train_labels = []
        test_datas = []
        test_labels = []
        cumulative_inc = 0  # 维护累计增量
        for taskid,data_flag in enumerate(self.domain_names):
            if taskid >= self.nb_sessions:
                    break
            current_inc = self.cl_n_inc[taskid]  # 当前session的增量
            info = INFO[data_flag]
            task = info['task']
            n_channels = info['n_channels']
            n_classes = len(info['label'])
            DataClass = getattr(medmnist, info['python_class'])
            train_dataset = DataClass(split='train', root='/Ds/xmu/mayuxi/CL/data/medmnist/data')
            val_dataset = DataClass(split='val', root='/Ds/xmu/mayuxi/CL/data/medmnist/data')
            test_dataset = DataClass(split='test', root='/Ds/xmu/mayuxi/CL/data/medmnist/data') 
            if n_channels == 1:
                train_dataset.imgs = np.repeat(train_dataset.imgs[..., np.newaxis], 3, axis=-1)# train_dataset.imgs.unsqueeze(1).repeat(1, 3, 1, 1)
                val_dataset.imgs = np.repeat(val_dataset.imgs[..., np.newaxis], 3, axis=-1)# val_dataset.imgs.unsqueeze(1).repeat(1, 3, 1, 1)
                test_dataset.imgs = np.repeat(test_dataset.imgs[..., np.newaxis], 3, axis=-1)# test_dataset.imgs.unsqueeze(1).repeat(1, 3, 1, 1)
            train_datas.append(train_dataset.imgs)
            train_labels.append(train_dataset.labels + cumulative_inc)
            train_datas.append(val_dataset.imgs)
            train_labels.append(val_dataset.labels + cumulative_inc)
            test_datas.append(test_dataset.imgs)
            test_labels.append(test_dataset.labels + cumulative_inc)
            # 更新累计增量
            cumulative_inc += current_inc
        
        self.train_data, self.train_targets = np.concatenate(train_datas, axis=0), np.concatenate(
            train_labels, axis=0
        ).flatten()
        self.test_data, self.test_targets = np.concatenate(test_datas, axis=0), np.concatenate(
            test_labels, axis=0
        ).flatten()


class iSKIN(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(sum(args.get("increment_per_session", [args["increment"]]*args["total_sessions"]))).tolist()
        self.class_order = class_order
        self.nb_sessions = args["total_sessions"]
        # 修改后：支持不同session的不同增量
        self.cl_n_inc = args.get("increment_per_session", [self.args["increment"]]*self.nb_sessions)
        # 校验增量配置长度
        if len(self.cl_n_inc) != self.nb_sessions:
            raise ValueError(f"increment_per_session长度({len(self.cl_n_inc)})与total_sessions({self.nb_sessions})不匹配")
        
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ["PH2", "MSK", "clinic_D7P", "derm_D7P", "HAM", "BCN", "OTHERS", "clinical", "dermoscopic", ]
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False

        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        def _read_data(image_list_paths) -> (np.ndarray, np.ndarray):
            imgs = []
            cumulative_inc = 0  # 维护累计增量
            for taskid, image_list_path in enumerate(image_list_paths):
                if taskid >= self.nb_sessions:
                    break
                current_inc = self.cl_n_inc[taskid]  # 当前session的增量
                with open(image_list_path) as f:
                    image_list = f.readlines()
                # 重写 target class := original value + taskid * args["increment"]
                for entry in image_list:
                    img_label = int(entry.split()[1])

                    # 类增量模式筛选
                    if self.class_incremental:
                        # 只保留当前增量区间内的样本
                        if not (0 <= img_label < current_inc):
                            continue
                    else:
                        # 域增量模式校验
                        if img_label >= current_inc:
                            raise ValueError(f"当前session增量数不足: {img_label} >= {current_inc}")
                    
                    # 标签重映射：原始标签 + 累计增量
                    remapped_label = img_label + cumulative_inc
                    imgs.append((entry.split()[0], remapped_label))
                
                cumulative_inc += current_inc  # 更新累计增量

            img_x, img_y = [], []
            for item in imgs:
                img_x.append(os.path.join(self.image_list_root, item[0]))
                img_y.append(item[1])

            return np.array(img_x), np.array(img_y)

        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        self.train_data, self.train_targets = _read_data(image_list_paths)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        self.test_data, self.test_targets = _read_data(image_list_paths)

class iCHEST(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(sum(args.get("increment_per_session", [args["increment"]]*args["total_sessions"]))).tolist()
        self.class_order = class_order
        self.nb_sessions = args["total_sessions"]
        # 修改后：支持不同session的不同增量
        self.cl_n_inc = args.get("increment_per_session", [self.args["increment"]]*self.nb_sessions)
        # 校验增量配置长度
        if len(self.cl_n_inc) != self.nb_sessions:
            raise ValueError(f"increment_per_session长度({len(self.cl_n_inc)})与total_sessions({self.nb_sessions})不匹配")
        
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ["PH2", "MSK", "clinic_D7P", "derm_D7P", "HAM", "BCN", "OTHERS", "clinical", "dermoscopic", ]
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False

        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        def _read_data(image_list_paths) -> (np.ndarray, np.ndarray):
            imgs = []
            cumulative_inc = 0  # 维护累计增量
            for taskid, image_list_path in enumerate(image_list_paths):
                if taskid >= self.nb_sessions:
                    break
                current_inc = self.cl_n_inc[taskid]  # 当前session的增量
                with open(image_list_path) as f:
                    image_list = f.readlines()
                for entry in image_list:
                    img_label = int(entry.split("--")[1])

                    # 类增量模式筛选
                    if self.class_incremental:
                        # 只保留当前增量区间内的样本
                        if not (0 <= img_label < current_inc):
                            continue
                    else:
                        # 域增量模式校验
                        if img_label >= current_inc:
                            raise ValueError(f"当前session增量数不足: {img_label} >= {current_inc}")
                    
                    # 标签重映射：原始标签 + 累计增量
                    remapped_label = img_label + cumulative_inc
                    imgs.append((entry.split("--")[0], remapped_label))
                
                cumulative_inc += current_inc  # 更新累计增量

            img_x, img_y = [], []
            for item in imgs:
                img_x.append(os.path.join(self.image_list_root, item[0]))
                img_y.append(item[1])

            return np.array(img_x), np.array(img_y)

        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        self.train_data, self.train_targets = _read_data(image_list_paths)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        self.test_data, self.test_targets = _read_data(image_list_paths)


class iOfficeHome(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = (np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.cl_n_inc = self.args["increment"]
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ['Art', "Clipart", "Product", "Real_World"]
        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        self.image_list_root = self.args["data_path"]

        image_list_paths = []

        for d in self.domain_names:
            write_domain_img_file2txt(self.image_list_root, d)
            split_domain_txt2txt(self.image_list_root, d, train_ratio=0.7, seed=self.args['seed'])

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]

        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            # imgs: (relative_path, label)
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
        test_x, test_y = [], []
        for item in imgs:
            test_x.append(os.path.join(self.image_list_root, item[0]))
            test_y.append(item[1])
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/imagenet-a/train/"
        test_dir = "/data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/cub/train/"
        test_dir = "/data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/omni/train/"
        test_dir = "/data/omni/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/vtab/train/"
        test_dir = "/data/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class food101(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(101).tolist()

    def download_data(self):
        train_dir = "/data/food101/train/"
        test_dir = "/data/food101/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, Resize, EnsureType

class iCystX(iData):
    use_path = True

    # 训练时的数据增强（3D版本）
    train_trsf = [
        LoadImage(image_only=True),             # 读取nii.gz
        EnsureChannelFirst(),                   # [D,H,W] → [1,D,H,W]
        ScaleIntensity(),                       # 灰度归一化到 [0, 1]
        Resize((112, 56, 32)),                    # 统一尺寸
        EnsureType(),                           # 保证为 torch.Tensor
    ]
    # 测试/验证时的变换
    test_trsf = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((112, 56, 32)),                    # 调整到统一尺寸
        EnsureType(),
    ]

    # 通用变换（通常不需要Normalize RGB通道）
    common_trsf = [    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(sum(args.get("increment_per_session", [args["increment"]]*args["total_sessions"]))).tolist()
        self.class_order = class_order
        self.nb_sessions = args["total_sessions"]

        # 修改后：支持不同session的不同增量
        self.cl_n_inc = args.get("increment_per_session", [self.args["increment"]]*self.nb_sessions)
        # 校验增量配置长度
        if len(self.cl_n_inc) != self.nb_sessions:
            raise ValueError(f"increment_per_session长度({len(self.cl_n_inc)})与total_sessions({self.nb_sessions})不匹配")
        
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ["NYU", "MCF", "NU", "AHN", "MCA", "IU", "EMC", ]
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False

        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):

        def _read_data(image_list_paths) -> (np.ndarray, np.ndarray):
            imgs = []
            cumulative_inc = 0  # 维护累计增量
            for taskid, image_list_path in enumerate(image_list_paths):
                if taskid >= self.nb_sessions:
                    break
                current_inc = self.cl_n_inc[taskid]  # 当前session的增量
                with open(image_list_path) as f:
                    image_list = f.readlines()
                # 重写 target class := original value + taskid * args["increment"]
                for entry in image_list:
                    img_label = int(entry.split()[1])

                    # 类增量模式筛选
                    if self.class_incremental:
                        # 只保留当前增量区间内的样本
                        if not (0 <= img_label < current_inc):
                            continue
                    else:
                        # 域增量模式校验
                        if img_label >= current_inc:
                            raise ValueError(f"当前session增量数不足: {img_label} >= {current_inc}")
                    
                    # 标签重映射：原始标签 + 累计增量
                    remapped_label = img_label + cumulative_inc
                    imgs.append((entry.split()[0], remapped_label))
                
                cumulative_inc += current_inc  # 更新累计增量

            img_x, img_y = [], []
            for item in imgs:
                img_x.append(os.path.join(self.image_list_root, item[0]))
                img_y.append(item[1])

            return np.array(img_x), np.array(img_y)

        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        self.train_data, self.train_targets = _read_data(image_list_paths)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        self.test_data, self.test_targets = _read_data(image_list_paths)
