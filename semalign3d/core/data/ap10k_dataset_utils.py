import numpy as np

AP10K_FAMILIES = {
    'Bovidae': [
        'antelope',#
        # 'argali sheep',
        'bison',#
        'buffalo',#
        'cow',
        'sheep'
    ],
    'Canidae': [
        'dog',
        'fox',
        'wolf'
    ],
    'Castoridae': ['beaver'],
    'Cercopithecidae': [
        'alouatta',
        'monkey',
        'noisy night monkey',
        'spider monkey',
        'uakari'
    ],
    'Cervidae': ['deer', 'moose'],
    'Cricetidae': ['hamster'],
    'Elephantidae': ['elephant'],
    'Equidae': ['horse', 'zebra'],#
    'Felidae': [
        'bobcat',
        'cat',
        'cheetah',
        'jaguar',
        # 'king cheetah',
        'leopard',
        'lion',
        'panther',
        'snow leopard',
        'tiger'
    ],
    'Giraffidae': ['giraffe'],
    'Hippopotamidae': ['hippo'],
    'Hominidae': ['chimpanzee', 'gorilla'],
    'Leporidae': ['rabbit'],
    'Mephitidae': ['skunk'],
    'Muridae': ['mouse', 'rat'],
    'Mustelidae': ['otter', 'weasel'],
    'Procyonidae': ['raccoon'],
    "Rhinocerotidae": ['rhino'],
    'Sciuridae': ['marmot', 'squirrel'],
    'Suidae': ['pig'],
    'Ursidae': ['brown bear', 'panda', 'polar bear'],  # 'black bear', 
}


# In total, there are 50 categories in the AP-10K dataset
AP10K_CATEGORIES = sum(AP10K_FAMILIES.values(), [])

AP10K_SUPER_CATEGORIES = list(AP10K_FAMILIES.keys())

# copied from utils_geoaware.py
AP10K_FLIP = [
    [0,1], # eye
    2, # nose
    3, # neck
    4, # root of tail
    [5,8], # shoulder
    [6,9], # elbow # knee
    [12, 15], # knee
    [7,10], # front paw 
    [13,16], # back paw
    [11,14], # hip
]

AP10K_KPT_INDEX_TO_FLIPPED_KPT_INDEX = -np.ones(17, dtype=int)
for kpt_index in AP10K_FLIP:
    if isinstance(kpt_index, list):
        kpt_idx_1, kpt_idx_2 = kpt_index
        AP10K_KPT_INDEX_TO_FLIPPED_KPT_INDEX[kpt_idx_1] = kpt_idx_2
        AP10K_KPT_INDEX_TO_FLIPPED_KPT_INDEX[kpt_idx_2] = kpt_idx_1
    else:
        AP10K_KPT_INDEX_TO_FLIPPED_KPT_INDEX[kpt_index] = kpt_index

def build_get_flipped_kpt_index():
    """ In AP10K, keypoints are the same for all categories.
    """
    def get_flipped_kpt_index(kpt_index):
        if AP10K_KPT_INDEX_TO_FLIPPED_KPT_INDEX[kpt_index] == -1:
            raise ValueError(f"Flipped kpt index not found for kpt index {kpt_index}")
        return AP10K_KPT_INDEX_TO_FLIPPED_KPT_INDEX[kpt_index]
    return get_flipped_kpt_index
