__author__ = 'cheverebe'

settings = {
    'name': 'ale1',
    'extension': 'png',
    'predefined_angle': None,
    'tolerance': 10,

    #'ksize': (50, 10),
    'blur_kernel_size': (5, 5),
    'min_size_factor': 20,
    'dil_erod_kernel_size': (8, 8),
    'dil_erod_kernel_size_segmentator': (8, 8),
    'dilation_kernel_size_shadow_mask': (8, 8),
    'region_distance_balance': 1,  # color/spatial
    'max_color_dist': 0.3,
    #METHODS
    # 0 BGR
    # 1 LAB
    # 2 HSV
    'method': 1

}

# BALCON - min angle found: 151
# PELOTA - min angle found: 109
#cono 166
#palmeras 124
# ale1: 97, 115(148) / 150
# ale2: 154, 179 / 123
# ale3: 148, 175 /
# auto: 148
# balcon: 149
# bird: 148
# forest1: 97, 107(148)
# forest2: 67-120(10??? NONE)
# kitti: 130(148)
# madera: 139(any)
# palmera: 123(any)
# pelota: 109~(90)
# r1: 106, 115(1-179)
# road2: 96, 137(148)
# road3: 154
# road4: 80(ANY)
# road5:147, 155(OK 156)
# road6: 146(117)