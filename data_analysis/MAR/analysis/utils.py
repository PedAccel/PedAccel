import os

from preprocessing import load_sickbay_data, load_accel_data, load_retro_data, load_mar_data, load_ecg_data

def validate_directory():
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    output_dir = os.path.join(os.path.dirname(os.getcwd()), "output")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return data_dir, output_dir

def validate_directory_pat_num(pat_num):
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    output_dir = os.path.join(os.path.dirname(os.getcwd()), "output")

    data_dir_pat_num = os.path.join(data_dir, f'Patient{pat_num}')
    output_dir_pat_num = os.path.join(output_dir, f'Patient{pat_num}')

    if not os.path.exists(data_dir_pat_num):
        os.makedirs(data_dir_pat_num)

    if not os.path.exists(output_dir_pat_num):
        os.makedirs(output_dir_pat_num)

    return data_dir_pat_num, output_dir_pat_num

def get_data(data_dir, pat_num, sickbay=False, accel=False, sbs=False, mar=False, ecg=False):
    sickbay_data, accel_data, sbs_data, mar_data, ecg_data = None, None, None, None, None

    if sickbay:
        sickbay_data = load_sickbay_data(data_dir, pat_num)
    if accel:
        accel_data = load_accel_data(data_dir, pat_num)
    if sbs:
        sbs_data = load_retro_data(data_dir, pat_num)
    if mar:
        mar_data = load_mar_data(data_dir, pat_num)
    if ecg:
        ecg_data = load_ecg_data(data_dir, pat_num)

    return sickbay_data, accel_data, sbs_data, mar_data, ecg_data



