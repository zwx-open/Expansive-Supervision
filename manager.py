from types import SimpleNamespace

STORE_TRUE = None
import os

DIV2K_TEST = "../../data/div2k/test_data"
DIV2K_TRAIN = "../../data/div2k/train_data"
KODAK = "../../data/Kodak"
TEXT_TEST = "../../data/text/test_data"
STANDFORD_XYZ = "../../data/stanford_shape_xyz"
DEMO_IMG = os.path.join(DIV2K_TEST, "00.png")

DEMO_DRAGON_SHAPE = os.path.join(STANDFORD_XYZ, "dragon.xyz")
DEMO_ARMA_SHAPE = os.path.join(STANDFORD_XYZ, "armadillo.xyz")


DEMO_AUDIO = "../../data/libri_test_clean_121726/121-121726-0000.flac"
LIBRI_SLICE = "../../data/libri_test_clean_121726"

class ParamManager(object):
    def __init__(self, **kw):
        self._tag = "exp"
        self.p = SimpleNamespace()
        self._exp = ""

        self._set_exp(**kw)

    def _set_default_parmas(self):
        self.p.model_type = "siren"
        self.p.input_path = DEMO_IMG
        self.p.eval_lpips = STORE_TRUE
        self.p.use_sampler = STORE_TRUE
        self.p.warm_up = 0

    def _set_exp(self, idx=0, exp_num="000"):
        self._set_default_parmas()
        self.exp_num = exp_num

        eval(f"self._set_exp_{exp_num}(idx)")

        # self.p.tag = f"{self._tag}_{self._exp}"
        self.p.tag = f"{self._exp}"
        self.p.lr = self._get_lr_by_model(self.p.model_type)
        self.p.up_folder_name = self._tag

    def _convert_dict_args_list(self):
        args_dic = vars(self.p)
        args_list = []
        for key, val in args_dic.items():
            args_list.append(f"--{key}")
            if val is not STORE_TRUE:
                args_list.append(str(val))
        self._print_args_list(args_list)
        return args_list

    def export_args_list(self):
        return self._convert_dict_args_list()

    def export_cmd_str(self, use_cuda=[0]):
        args_list = self._convert_dict_args_list()
        script = "python main.py " + " ".join(args_list)
        script = self.add_cuda_visible_to_script(script, use_cuda)
        return script

    @staticmethod
    def add_cuda_visible_to_script(script, use_cuda=[0]):
        visible_devices: str = ",".join(map(str, use_cuda))
        return f"CUDA_VISIBLE_DEVICES={visible_devices} {script}"

    @staticmethod
    def _print_args_list(args_list):
        print("#" * 10 + "print for vscode debugger" + "#" * 10)
        for item in args_list:
            print(f'"{item}",')

    def _get_lr_by_model(self, model):
        if model == "gauss" or model == "wire":
            return 5e-3
        elif model == "siren":
            return 1e-4  # 1e-4 | 5e-4
        elif model == "finer":
            return 5e-4
        elif model == "pemlp":
            return 1e-3
        else:
            raise NotImplementedError

    def _use_single_data(self, pic_index="02", datasets = DIV2K_TEST):
        if hasattr(self.p, "multi_data"):
            delattr(self.p, "multi_data")

        self.p.input_path = os.path.join(datasets, f"{pic_index}.png")
        self._tag += f"_single_{pic_index}"

    def _use_datasets(self, type="div2k_test"):
        self.p.multi_data = STORE_TRUE
        if type == "div2k_test":
            self.p.input_path = DIV2K_TEST
        elif type == "div2k_train":
            self.p.input_path = DIV2K_TRAIN
        elif type == "kodak":
            self.p.input_path = KODAK
        elif type == "text_test":
            self.p.input_path = TEXT_TEST
        elif type == "libri_slice":
            self.p.input_path = LIBRI_SLICE
        self._tag += f"_{type}"

    ##############

    def _set_exp_000(self, idx):
        self._tag = "000_examples"
        self._exp_name_list = [
            "model-siren",
            "model-finer",
            "model-pemlp",
            "model-gauss",
            "model-wire",
        ]
        self._exp = self._exp_name_list[idx]
        self.p.num_epochs = 10
        if idx == 0:
            self.p.model_type = "siren"
        elif idx == 1:
            self.p.model_type = "finer"
        elif idx == 2:
            self.p.model_type = "pemlp"
        elif idx == 3:
            self.p.model_type = "gauss"
        elif idx == 4:
            self.p.model_type = "wire"
        else:
            raise NotImplementedError

    def _set_exp_001(self, idx):
        self._set_exp_000(idx)
        self._tag = "001_examples_multi"
        self.p.input_path = DIV2K_TEST
        self.p.multi_data = STORE_TRUE
        self.p.num_epochs = 10
        self.p.log_epoch = 5

    def _set_exp_002(self, idx):
        self._tag = "002_trans"
        self._exp_name_list = [
            "min_max",
            "z_score",
            "sym_power",
        ]
        self._exp = self._exp_name_list[idx]

        if idx == 0:
            self.p.transform = "min_max"
        elif idx == 1:
            self.p.transform = "z_score"
        elif idx == 2:
            self.p.transform = "sym_power"
        else:
            raise NotImplementedError
        # self.p.num_epochs = 500
        # self.p.log_epoch = 100

        self.p.input_path = DIV2K_TEST
        self.p.multi_data = STORE_TRUE

    def _set_exp_003(self, idx):
        self._tag = "003_loss_components"
        self._exp_name_list = [
            "mse",
            "laplacine_1e-5",
            "laplacine_1e-3",
            "laplacine_1e-4",
            "laplacine_1e-6",
            "laplacine_5e-5",
            "use_blur",
        ]
        self._exp = self._exp_name_list[idx]
        if idx == 0:
            pass
        elif idx == 1:
            self.p.lambda_l = "1e-5"
        elif idx == 2:
            self.p.lambda_l = "1e-3"
        elif idx == 3:
            self.p.lambda_l = "1e-4"
        elif idx == 4:
            self.p.lambda_l = "1e-6"
        elif idx == 5:
            self.p.lambda_l = "5e-5"
        elif idx == 6:
            self.p.use_blur_sup = STORE_TRUE
        else:
            self.p.lambda_l = idx
            self._exp = f"laplacine_{self.p.lambda_l}"

        self.p.input_path = DIV2K_TEST
        self.p.multi_data = STORE_TRUE

    def _set_exp_004(self, idx):
        self._set_exp_000(idx)
        self._tag = "004_one_forward"
        self.p.num_epochs = -1
        self.p.input_path = DEMO_IMG
        self.p.hidden_layers = 3
        self.p.hidden_features = 256

    def _set_exp_005(self, idx):
        """初步结果:简单的线性动态调控不如定死"""
        self._tag = "005_lap_lamda"
        self.p.input_path = DEMO_IMG
        self._exp_name_list = [
            "mse",
            "laplacine_1e-5",
        ]
        self._exp = self._exp_name_list[idx]
        if idx == 0:
            pass
        elif idx == 1:
            self.p.lambda_l = "1e-5"
        else:
            self.p.lambda_l = "1e-5"
            self.p.lambda_l_schedular = idx
            self._exp = f"{self.p.lambda_l}_move_{self.p.lambda_l_schedular}"

    def _set_exp_006(self, _exp="baseline"):
        """bias | 探究不同lamda系数下的作用(0阶、1阶、2阶)"""
        self._tag = "006_spectral_bias"
        self.p.num_epochs = 500
        self.p.log_epoch = 50
        self.p.signal_type = "series"
        self.p.hidden_layers = 1
        self.p.hidden_features = 64

        self._exp = _exp

        if _exp == "baseline":
            pass
        else:
            l0, l1, l2 = _exp.split("_")
            self.p.lamda_0 = l0
            self.p.lamda_1 = l1
            self.p.lamda_2 = l2

        #  param_idxs:list = ["baseline","0_1_0", "0_0_1", "1_1_1",]
        #  param_idxs:list = ["1_0.5_0.5","1_2_2", "1_0.453_0.158"]
        # param_idxs:list = ["1_5_5","1_10_10", "1_2_5", "1_5_2"]

    def _set_exp_007(self, _exp="baseline"):
        """spectral_bias | 探究不同阶数情况下收敛情况, lamda均为1"""
        self._tag = "007_spectral_bias"
        self.p.num_epochs = 500
        self.p.log_epoch = 50
        self.p.signal_type = "series"
        self.p.hidden_layers = 1
        self.p.hidden_features = 64

        self.p.lamda_0 = 1
        self.p.lamda_1 = 1
        self.p.lamda_2 = 1

        self._exp = f"order_{_exp}"
        self.p.order_sup = _exp

    def _set_exp_008(self, _exp):
        """改了一下数值稳定性做初始化 | 从数值稳定性上看确实阶数越高收敛越好(越高阶收益越小)"""
        """siren越高阶越好 pemlp一阶就好 wire就到两阶"""
        """其实siren原paper有说明,目前打算把这个增益融到frozen这篇里"""
        self._set_exp_007(_exp)
        self._tag = "008_spectral_bias_pemlp"
        self.p.hidden_layers = 1
        self.p.hidden_features = 32
        self.p.model_type = "pemlp"

    def _set_exp_009(self, _exp):
        """测试img模态loss重心偏移是否有影响"""
        """DEMO img上加速收敛了一下 但是最终还是不如vanilla"""
        """用laplace做先验也不如vanilla"""
        self._tag = "009_important_loss_lap"
        self._exp = _exp
        self.p.input_path = DEMO_IMG
        self.p.lamda_1 = 1
        self.p.ratio = _exp

    def _set_exp_010(self, _exp):
        """example for single"""
        self._tag = "010_examples"
        self._exp = "error_map"
        self.p.log_epoch = 500
        self.p.num_epochs = 5000

    def _set_exp_012(self, _exp):
        self._tag = "012_sampling_dev_0.3"
        """看起来freeze+melt能让一开始收敛快速(1k-2k), 但是后续比不过full ratio(5k)"""

        self.p.log_epoch = 500
        self.p.num_epochs = 2000

        self.p.use_sampler = STORE_TRUE

        use_ratio = 0.3

        self.p.strategy = _exp
        self.p.use_ratio = use_ratio

        # freeze_1
        self.p.warm_up = 0  # 这个倒是无妨 | 还是有的话效果明显一点儿
        self.p.init_interval = 50

        self._exp = f"{_exp}_ratio_{use_ratio}"

        if _exp.startswith("melt"):
            self.p.strategy = "freeze"
            melt = float(_exp.split("_")[1])
            self.p.melt_ratio = melt
            # self.p.melt_ratio = 0.5 * use_ratio
            self.p.freeze_ratio = 1 - use_ratio + self.p.melt_ratio
            # 变化率
            # self.p.profile_tool = "error_rate"

    def _set_exp_0121(self, _exp):
        self._set_exp_012(_exp)
        self._tag = "012_sampling_natural_test"
        self.p.input_path = DIV2K_TEST
        self.p.multi_data = STORE_TRUE
        self.p.log_epoch = 500
        self.p.num_epochs = 5000

    def _set_exp_013(self, _exp):

        use_ratio = 0.5
        self._tag = f"013_sampling_strategy_{use_ratio}"
        self._exp = f"{use_ratio}-{_exp}"

        """固定pixel_rate_ 正式把所有sampling baseline都跑一下"""
        self.p.log_epoch = 500
        self.p.num_epochs = 2000

        self.p.use_ratio = use_ratio
        # self.p.init_interval = 50

        if _exp == "full":
            self.p.strategy = "full"
        elif _exp == "full_lap":
            self.p.strategy = "full"
            self.p.lap_coff = 1e-5
        elif _exp == "random":
            self.p.strategy = "random"
        elif _exp == "freeze":
            self.p.strategy = "freeze"
            self.p.init_mutation_ratio = 0
        elif _exp == "fm":
            """默认用 0.5 * use_ratio"""
            self.p.strategy = "freeze"

        elif _exp == "fm_diff_1":
            self.p.strategy = "freeze"
            self.p.profile_guide = "diff_1"

        elif _exp == "fm_linear_dec_prof":
            self.p.strategy = "freeze"

            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 100

        elif _exp == "fm_laplace":
            self.p.strategy = "freeze"

            self.p.lap_coff = 1e-5

        elif _exp == "fm_laplace_dec":
            self.p.strategy = "freeze"

            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 100

            self.p.lap_coff = 1e-5

        elif _exp == "fm_cross_laplace_dec":
            self.p.strategy = "freeze"

            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 75

            self.p.lap_coff = 1e-5
            self.p.crossover_method = "add"

        elif _exp == "fm_cross_laplace_fixed":
            self.p.strategy = "freeze"

            self.p.lap_coff = 1e-5
            self.p.crossover_method = "add"
            self.p.init_interval = 25

        elif _exp == "fm_final":
            ##### 最终选取还不错的版本
            self.p.strategy = "freeze"

            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 75

            self.p.lap_coff = 1e-5
            self.p.crossover_method = "add"
            self.p.mutation_method = "linear"

        # elif _exp.startswith("melt"):
        #     self.p.strategy = "freeze"
        #     melt = float(_exp.split("_")[1])
        #     self.p.melt_ratio = melt
        #     self.p.freeze_ratio = 1 - use_ratio + self.p.melt_ratio

        elif _exp == "nmt_incre":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "incremental"
        elif _exp == "nmt_dense2":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense2"
        elif _exp == "nmt_dense":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense"
        elif _exp == "nmt_rev_incre":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "reverse-incremental"
        elif _exp == "soft":
            self.p.strategy = "soft"

        # elif _exp == "dev":
        #     self._tag = f"dev_sampling_strategy_{use_ratio}"
        #     self._exp = f"dev-xx-{use_ratio}"

        #     self.p.strategy = "freeze"

        #     # todo:
        #     self.p.eval_lpips = STORE_TRUE

    def _set_exp_0131(self, _exp):
        self._set_exp_013(_exp)
        self.p.input_path = DIV2K_TEST
        self.p.multi_data = STORE_TRUE
        self.p.log_epoch = 500
        self.p.num_epochs = 5000
        self._tag = f"013_sampling_strategy_div2k_test"

    def _set_exp_014(self, interval):
        """for measurement mdd l1前后变换的diff"""
        pic_index = "02"
        self._tag = f"014_measurement_supp"
        self._exp = f"{pic_index}_{interval}"
        self.p.log_epoch = 5000
        self.p.num_epochs = 5000
        self.p.use_sampler = STORE_TRUE
        self.p.strategy = "full"
        self.p.mdd_interval = interval

        self.p.input_path = os.path.join(DIV2K_TEST, f"{pic_index}.png")
        self.p.measure_dense_diff = STORE_TRUE

    def _set_exp_015(self, para):
        """测试interval fixed or linear decrease"""
        # fm_组合
        self._tag = f"015"
        self._exp = f"{para}"

        use_ratio = 0.5
        self.p.use_ratio = use_ratio
        self.p.strategy = "freeze"
        # self.p.melt_ratio = 0.5 * use_ratio
        # self.p.freeze_ratio = 1 - use_ratio + self.p.melt_ratio
        self.p.use_sampler = STORE_TRUE
        self.p.eval_lpips = STORE_TRUE

        pic_index = "02"
        self.p.input_path = os.path.join(DIV2K_TEST, f"{pic_index}.png")

        if para == "fixed":
            self.p.profile_interval_method = para
            self.p.init_interval = 50
        elif para == "lin_dec":
            self.p.profile_interval_method = para
            self.p.init_interval = 100

    def _set_exp_016(self, pic_index="02"):
        """每一轮 l1 和 lap的diff"""
        self._tag = f"016_measurement"
        self._exp = f"{pic_index}"
        self.p.log_epoch = 5000
        self.p.num_epochs = 5000
        self.p.use_sampler = STORE_TRUE
        self.p.strategy = "full"

        self.p.input_path = os.path.join(DIV2K_TEST, f"{pic_index}.png")
        self.p.measure_crossover_diff = STORE_TRUE

    def _set_exp_017(self, param):
        """测试laplace loss 在sampling时的作用"""
        """full_1e-5"""
        pic_index = "02"
        self._tag = f"{self.exp_num}"
        self._exp = f"{param}"
        self.p.log_epoch = 500
        self.p.num_epochs = 5000
        self.p.strategy = "full"

        self.p.input_path = os.path.join(DIV2K_TEST, f"{pic_index}.png")

        mode, lap_coff = param.split("_")
        self.p.strategy = mode
        self.p.lap_coff = float(lap_coff)

        if mode == "fm":
            self.p.use_ratio = 0.5
            self.p.strategy = "freeze"

            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 100

    def _set_exp_018(self, param):
        """cross_over"""
        pic_index = "02"
        self._tag = f"{self.exp_num}"
        self._exp = f"{param}"
        self.p.input_path = os.path.join(DIV2K_TEST, f"{pic_index}.png")

        ### fm all designs
        self.p.use_ratio = 0.5
        self.p.strategy = "freeze"

        self.p.profile_interval_method = "lin_dec"
        self.p.init_interval = 75

        self.p.lap_coff = 1e-5

        ### crossover
        self.p.crossover_method = param

    def _set_exp_019(self, param):
        """mutation"""
        self._set_exp_018(param)
        self.p.crossover_method = "add"

        self.p.mutation_method = "exp"
        self.p.init_mutation_ratio = param
        self.p.end_mutation_ratio = 0.4
        self._exp = f"{self.p.mutation_method}_{param}_{self.p.end_mutation_ratio}"

    def _set_fm_final(self):
        self.p.strategy = "freeze"
        # profile
        self.p.profile_interval_method = "lin_dec"
        self.p.init_interval = 75
        self.p.end_interval = 50
        # crossover
        self.p.lap_coff = 1e-5
        self.p.crossover_method = "add"
        # mutation
        self.p.mutation_method = "exp"
        self.p.init_mutation_ratio = 0.6
        self.p.end_mutation_ratio = 0.4

    def _set_fm_final2(self):
        self.p.strategy = "freeze"
        # profile
        self.p.profile_interval_method = "lin_dec"
        # self.p.init_interval = 75
        self.p.init_interval = 100
        self.p.end_interval = 50
        # crossover
        self.p.lap_coff = 1e-5
        self.p.crossover_method = "select"
        # mutation
        self.p.mutation_method = "constant"
        self.p.init_mutation_ratio = 0.5
        # self.p.init_mutation_ratio = 0.6
        # self.p.end_mutation_ratio = 0.4


    def _set_exp_020(self, _exp):
        """正式开始测试策略 → div2k"""

        use_ratio = 0.5

        self._tag = f"120_final_{use_ratio}" # 120 represent formal
        self._exp = f"{_exp}"

        self.p.log_epoch = 500 
        self.p.num_epochs = 5000
        self.p.use_ratio = use_ratio

        # self._use_single_data("02")
        self._use_datasets()

        self._by_sampler(_exp)
    
    def _set_exp_02020(self, _exp):
        '''测target psnr的时间 dense profile'''
        self._set_exp_020(_exp)
        self._tag = f"02020_target_PSNR"
        self.p.log_epoch = 50
        self._use_datasets()


    def _set_exp_0201(self, _exp):
        '''ablation'''
        self._set_exp_020(_exp)
        use_ratio = 0.5
        self._tag = f"020_final_{use_ratio}_ablation"
        self._use_datasets()


    def _set_exp_021(self, _exp):
        """测试动态nmt的策略"""
        use_ratio = 0.2
        self._set_exp_020(_exp)
        self.p.use_ratio = use_ratio

        self.p.sample_num_schedular = "step" # constant", "linear", "step", "cosine", "reverse-cosine"

        self._tag = f"121_schduler_{use_ratio}_{ self.p.sample_num_schedular}"
        # datasets
        self._use_datasets()

    def _set_exp_022(self, _exp):
        use_ratio = 0.5
        self.p.record_indices = STORE_TRUE
        self._set_exp_020(_exp)
        self.p.use_ratio = use_ratio
        self._tag = f"022_visualize_{use_ratio}"

        ## step
        self.p.use_ratio = 0.2
        self.p.sample_num_schedular = "step"
        self._tag = f"022_visualize_{use_ratio}_step"

        self._use_single_data("03")
    
    def _set_exp_023(self, _exp):
        '''different backbone'''

        use_ratio = 0.5

        self._tag = f"023_backbone_{use_ratio}"
        self._exp = f"{_exp}"

        self.p.log_epoch = 500
        self.p.num_epochs = 5000
        self.p.use_ratio = use_ratio
        
        # use learning rate scheduler ### 测试wire和gauss需要加上 否则不work
        self.p.use_lr_scheduler = STORE_TRUE

        self._use_datasets("kodak")

        _backbone, _st = _exp.split("_")
        

        if _st == "full":
            self.p.strategy = "full"

        elif _st == "random":
            self.p.strategy = "random"
    
        elif _st == "fm":
            self._set_fm_final()
        
        self.p.model_type = _backbone

        # step
        # self.p.use_ratio = 0.2
        # self.p.sample_num_schedular = "step"
        # self._exp += "_step"
        
        
        # tag_list = []
        # for backbone in ["siren", "finer", "wire", "guass","pemlp"]:
        #     for st in ["full", "fm"]:
        #         tag_list.append(f"{backbone}_{st}")
        # print(tag_list)
    
    def _set_exp_024(self, _exp):
        """输出单图的各个pred图像, 可视化用"""
        self._set_exp_020(_exp)
        img_id = "06"
        self._tag = f"024_visual_compare_{img_id}"
        self.p.snap_epoch = 500
        self._use_single_data(img_id)
    
    def _set_exp_025(self, _exp):
        """dense measure psnr 用于可视化loss曲线"""
        self._set_exp_020(_exp)
        img_id = "06"
        self._tag = f"025_dense_measure_loss_{img_id}"
        self.p.dense_measure_psnr = STORE_TRUE
        self._use_single_data(img_id)

    def _set_exp_026(self, _exp):
        """输出单图的各个pred图像(backbone), 可视化用"""
        self._set_exp_023(_exp)
        img_id = "23"
        self._tag = f"026_visual_compare_{img_id}"
        self.p.snap_epoch = 100
        self._use_single_data(img_id, datasets=KODAK)

    def _set_exp_027(self, _exp):
        """测试不同network size的效果"""
        
        use_ratio = 0.5

        self._tag = f"027_1w_nwteork_size_{use_ratio}"
        self._exp = f"{_exp}"

        self.p.log_epoch = 500
        self.p.num_epochs = 10000
        self.p.use_ratio = use_ratio

        _features, _layers,  _st = _exp.split("_")
        self.p.hidden_layers = _layers
        self.p.hidden_features = _features
        

        if _st == "full":
            self.p.strategy = "full"

        elif _st == "random":
            self.p.strategy = "random"
    
        elif _st == "fm":
            self._set_fm_final()
            
        elif _st == "fmstep":
            self._set_fm_final()
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"

        
        self._use_datasets("div2k_test")
        
        # tag_list = []
        # for feat in ["64","128","256"]:
        #     for layer in ["1","2","3"]:
        #         for st in ["full", "fm"]:
        #             tag_list.append(f"{feat}_{layer}_{st}")
        # print(tag_list)

    def _set_exp_030(self, _exp):
        '''Modality: Text Image'''
        self._set_exp_020(_exp)
        self._tag = f"030_text_img_sample_methods_finer"

        self.p.log_epoch = 50
        self.p.snap_epoch = 50
        self._tag += "_snap"
        self.p.num_epochs = 1000
        self._use_datasets("text_test")

    def _set_exp_031(self, _exp):
        '''different backbone for text data'''
        _backbone, _sample_method = _exp.split("+")
        self._set_exp_020(_sample_method)
        self.p.model_type = _backbone

        self.p.log_epoch = 50
        self.p.snap_epoch = 50
        self.p.num_epochs = 1000
        self._use_datasets("text_test")
        self._tag = f"031_text_img_backbone"
        self._exp = f"{_exp}"

    
    def _set_exp_040(self, _exp):
        """ Modality: Audio"""
        
        self.p.signal_type = "shape"
        self.p.input_path = DEMO_ARMA_SHAPE
        self.p.log_epoch = 1000
        self.p.num_epochs = 10000
        self.p.hidden_layers = 8
        self.p.hidden_features = 256

        # sampler
        self._by_sampler(_exp, signal="shape")
        
        self._tag = f"040_sdf_sampler_ARMA"
        self._exp = f"{_exp}"

    def _set_exp_050(self, _exp):
        """Modality : Audio"""
        self.p.signal_type = "audio"
        self.p.input_path = DEMO_AUDIO
        self._use_datasets("libri_slice")
        
        self.p.log_epoch = 1000 
        self.p.num_epochs = 5000

        ### adjust param for SIREN
        self.p.first_omega = 100
        self.p.hidden_omega = 100

        self._by_sampler(_exp, signal="audio")

        self._tag = f"050_audio_sampler"
        self._exp = f"{_exp}"
    

    def _set_exp_060(self, _exp):
        """Modality : NERF"""
        self.p.signal_type = "radiance_field"
        self.p.nerf_dataset = "syn"
        self.p.nerf_scene = "lego"
        self.p.nerf_backbone = "tensorf"

        self._tag = f"060_nerf_vram"
        # self._tag = f"060_nerf_lego_time"
        # self._tag = f"060_nerf_ship"
        self._exp = f"{_exp}"
        
        if _exp == "soft":
            self.p.strategy = "soft"
        elif _exp == "full":
            self.p.strategy = "full"
        elif _exp == "expansive":
            self.p.strategy = "expansive"
        elif _exp == "egra":
            self.p.strategy = "egra"
        elif _exp == "expansive_0.5":
            self.p.strategy = "expansive"
            self.p.batch_size = int(0.5 * 4096)
        elif _exp == "expansive_0.7":
            self.p.strategy = "expansive"
            self.p.batch_size = int(0.7 * 4096)
        else:
            self.p.strategy = "expansive"
            self.p.batch_size = int(float(_exp.split("_")[1]) * 4096)
        
        # multidatasets
        # self.p.multi_data = STORE_TRUE
    
    def _set_exp_0601(self, _exp): 
        self._set_exp_060(_exp)
        self._tag = f"0601_nerf_llff"
        self.p.nerf_dataset = "llff"
        self.p.soft_mining_alpha = 0.8
        

    def _set_exp_061(self, _exp):
        '''test expansive supervision / ablation'''
        self.p.signal_type = "radiance_field"
        self.p.nerf_dataset = "syn"
        self.p.multi_data = STORE_TRUE
        self.p.nerf_backbone = "tensorf"
        self.p.strategy = "full"
        
        ### batch size
        self._tag = f"061_nerf_es_test"
        self._exp = f"{_exp}"
        self.p.batch_size = int(float(_exp.split("_")[1]) * 4096)

        ### alation
        # self._tag = f"061_nerf_es_ablation"
        # self._exp = f"{_exp}"
        # if _exp == "wo_anchor_sample":
        #     self.p.es_beta_a = 0.0
        # elif _exp == "wo_source_sample":
        #     self.p.es_beta_a = 1.0
        # elif _exp == "wo_anchor_sup":
        #     self.p.lamda_a = 0.
        # elif _exp == "wo_source_sup":
        #     self.p.lamda_s = 0.
        # elif _exp == "a_s_add":
        #      self.p.a_s_add = STORE_TRUE
        # elif _exp == "use_mse":
        #     self.p.use_mse = STORE_TRUE



    def _set_exp_062(self, _exp):
        '''test soft mining alpha
        syn: alpha=0.4
        '''
        self.p.signal_type = "radiance_field"
        self.p.nerf_dataset = "syn"
        self.p.nerf_backbone = "tensorf"

        self.p.multi_data = STORE_TRUE
        self._tag = f"062_soft_tune"
        self._exp = f"{_exp}"
        self.p.strategy = "soft"
        self.p.soft_mining_alpha = float(_exp.split("_")[1])

    



    def _set_exp_dev(self, _exp):
        use_ratio = 0.5
        self._tag = f"dev"
        self._exp = f"dev_{_exp}"

        self.p.log_epoch = 500
        self.p.num_epochs = 5000
        self.p.use_ratio = use_ratio
        
        # self.p.strategy = "full" # "soft" "random"
        self._set_fm_final2()
        
        # self.p.measure_sample_diff = STORE_TRUE
        # self.p.soft_raw = STORE_TRUE
        
        # self.p.wo_correction_loss = STORE_TRUE



        ## single data
        self._use_single_data("03")
        # self.p.record_indices = STORE_TRUE
        # self._use_datasets()        
        # self._set_fm_final()

        ## test expansive
        # self.p.strategy = "freeze"
        # # crossover
        self.p.lap_coff = 1e-5
        # self.p.crossover_method = "add"
        # # self.p.sample_num_schedular = "step"

    
    def _set_evos_args(self, signal):
        self.p.strategy = "evos"
        if signal == "shape":
            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 100
            self.p.end_interval = 50
            self.p.lap_coff = 0
            self.p.crossover_method = "no"
            self.p.use_laplace_epoch = 0
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = 0.5

        elif signal == "audio":
            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 100
            self.p.end_interval = 50
            self.p.lap_coff = 0
            self.p.crossover_method = "no"
            self.p.use_laplace_epoch = 0
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = 0.5

        elif signal == "image":
            self.p.profile_interval_method = "lin_dec"
            self.p.init_interval = 100
            self.p.end_interval = 50
            # crossover
            self.p.lap_coff = 1e-5
            self.p.crossover_method = "select"
            self.p.use_laplace_epoch = 1500
            # mutation
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = 0.5

    
    def _by_sampler(self, _exp, signal="image"):

        if _exp == "full":
            self.p.strategy = "full"

        elif _exp == "random":
            self.p.strategy = "random"
        
        elif _exp == "evos":
            self.p.strategy = "evos"
            self._set_evos_args(signal) # same args
        
        elif _exp == "evos_wo_laplace":
            self.p.strategy = "evos"
            self._set_evos_args(signal) # same args
            self.p.lap_coff = 0
        
        elif _exp == "evos_step":
            self.p.strategy = "evos"
            self._set_evos_args(signal) # same args
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"

        elif _exp == "fm_cur":
            self._set_fm_final()
        
        elif _exp == "fm_cur2":
            self._set_fm_final2()
        
        elif _exp == "fm_cur2_step":
            self._set_fm_final2()
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"
        
        elif _exp == "fm_cur2_wo_crossover_laploss":
            self._set_fm_final2()
            self.p.lap_coff = 0

        elif _exp == "fm_cur_fix_mutation":
            self._set_fm_final()
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = 0.6

        elif _exp == "fm_cur_fix_profile":
            self._set_fm_final()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = 75
        
        elif _exp.startswith("fm_cur2_fix_profile"):
            interval = _exp.split("_")[-1]
            self._set_fm_final2()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = interval

        elif _exp == "fm_cur_wo_crossover_all":
            self._set_fm_final()
            self.p.lap_coff = 0
            self.p.crossover_method = "no"

        elif _exp == "fm_cur_wo_crossover_laploss":
            self._set_fm_final()
            self.p.lap_coff = 0

        elif _exp == "fm_cur_wo_crossover_add":
            self._set_fm_final()
            self.p.crossover_method = "no"

        elif _exp == "fm_cur_wo_mutation":
            self._set_fm_final()
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = 0
        
        elif _exp.startswith("fm_cur2_fix_mutation"):
            ratio = _exp.split("_")[-1]
            self._set_fm_final2()
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = ratio

        elif _exp == "fm_cur_wo_profile":
            self._set_fm_final()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = self.p.num_epochs

        elif _exp == "fm_cur_dense_profile":
            self._set_fm_final()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = 2

        # nmt
        elif _exp == "nmt_incre":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "incremental"
        elif _exp == "nmt_incre_step":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "incremental"
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"

        elif _exp == "nmt_dense2":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense2"
        elif _exp == "nmt_dense":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense"

        elif _exp == "nmt_dense_step":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense"
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"
        
        elif _exp == "nmt_rev_incre":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "reverse-incremental"

        # soft mining
        elif _exp == "soft":
            self.p.strategy = "soft"
        
        elif _exp == "soft_hard":
            self.p.strategy = "soft"
            self.p.soft_mining_alpha = 0
        
        elif _exp == "soft_imp":
            self.p.strategy = "soft"
            self.p.soft_mining_alpha = 1

        elif _exp == "soft_mse":
            self.p.strategy = "soft"
            self.p.wo_correction_loss = STORE_TRUE
        
        elif _exp == "soft_raw":
            self.p.strategy = "soft"
            self.p.soft_raw =  STORE_TRUE

        # expansive supervision
        elif _exp == "expansive":
            self.p.strategy = "expansive"
        
        elif _exp == "egra":
            self.p.strategy = "egra"

        #### fintune
        elif _exp.startswith("tune_alpha"):
            parmas = _exp.split("_")
            self._set_fm_final()
            _alpha = parmas[-1]
            self.p.strategy = "soft"
            self.p.soft_mining_alpha = _alpha
            # self.p.mutation_method = "constant"
            # self.p.init_mutation_ratio = parmas[1]
            # self.p.init_interval = parmas[1]
            # self.p.end_interval = parmas[2]