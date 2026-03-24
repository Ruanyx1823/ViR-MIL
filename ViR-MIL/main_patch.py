"""
ViLa-MIL主程序补丁文件
将此文件中的代码添加到原始main.py文件中的适当位置
"""

# 在main.py中添加以下代码，位置在其他任务类型判断之后（约在第104行）

"""
elif args.task == 'task_vitiligo_subtyping':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/vitiligo_subtyping.csv',
                                mode = args.mode,
                                data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                shuffle = False,
                                print_info = True,
                                label_dict = {'Stable': 0, 'Developing': 1},
                                patient_strat= False,
                                ignore=[])
"""