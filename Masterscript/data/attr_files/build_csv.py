import csv
import numpy as np
import itertools
import pandas as pd
import math

WIDTH_MIN = 0.1 # width of shadow from perspective of vehicle (m)
WIDTH_MAX = 4

LENGTH_MIN = 1 # length of shadow from perspective of vehicle (m)
LENGTH_MAX = 30

BETA_MIN = 0 # rotation of negative shadow relative to lane marker (degrees)
BETA_MAX = 90

TRANSPARENCY_MIN = 90 # transparency of positive shadow
TRANSPARENCY_MAX = 90

BLUR_MIN = 0 # degree of blur (softness) of positive shadow
BLUR_MAX = 0

DISTANCE_MIN = 0.1 # distance of negative shadow relative to lane marker (m)
DISTANCE_MAX = 4

DIV_NUM = 25 # number of rows per csv
MAX_X = 4
def make_list(min: float, max: float, step: float) -> list:
    """ Creates a list based on input params, rounds to second decimal place
    """
    return [round(x, 2) for x in np.arange(min, max, step)]

def write_csv(rows, num):
        
    with open(f'shadow_attrs{num}.csv', mode='w') as attr_file:
        attr_writer = csv.writer(attr_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        title_row = ['width', 'length', 'beta', 'transparency', 'blur', 'distance']
        attr_writer.writerow(title_row)

        for row in rows:
            attr_writer.writerow(row)

def find_x2(row):
    """
    """
    distance = row['distance']
    width = row['width']
    length = row['length']
    beta = math.radians(90 - row['beta'])

    return distance + width * math.sin(beta) + length * math.cos(beta)


def build_rows():
    width_list = make_list(WIDTH_MIN, WIDTH_MAX, 0.2)
    length_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 40]
    beta_list = [0, 5, 10, 15, 30, 45, 90]
    transparency_list = [90]
    blur_list = [0]
    distance_list = make_list(DISTANCE_MIN, DISTANCE_MAX, 0.2)

    print('Building combinations list...')
    all_combinations = list(itertools.product(width_list, length_list, beta_list, \
                                              transparency_list, blur_list, distance_list))
    df = pd.DataFrame(all_combinations, columns=['width', 'length', 'beta', 'transparency', 'blur', 'distance'])
    df['ns_width'] = df.apply(find_x2, axis=1)
    df = df.drop(df[(df['ns_width'] > MAX_X)].index)
    print(df.shape[0])
    len_comb = df.shape[0]
    num_img = 4 * len_comb * 3
    mem_tot = round((num_img * 871.7) * 1e-6, 2)
    num_csv = (len_comb // DIV_NUM) + 1 if (len_comb % DIV_NUM != 0) else 0
    csv_mem = round((DIV_NUM * 4 * 3 * 871.7) * 1e-6, 2)
    
    print(f'List with {len_comb} rows built')
    print(f'Will result in {num_img} images')
    print(f'Will take up total {mem_tot} GB')
    print(f'If dividing into {DIV_NUM} row csv\'s, will need {num_csv} files, with each taking up {csv_mem} GB')
    df.drop('ns_width', inplace=True, axis=1)

    return df, num_csv

def divide_files(shadows, num_csv):
    for i in range(num_csv):
        sub_list = shadows.iloc[i*DIV_NUM: (i+1)*DIV_NUM]
        sub_list.to_csv(f'shadow_attrs{i}.csv', index=False)


if __name__ == '__main__':
    total_list, num_csv = build_rows()
    divide_files(total_list, num_csv)
