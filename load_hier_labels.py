import pickle as pickle

train_test_info = open('./CUB_200_2011_train_test_multi_level_info.pkl', 'rb')
filelist_train = pickle.load(train_test_info)
label_train = pickle.load(train_test_info)
filelist_test = pickle.load(train_test_info)
label_test = pickle.load(train_test_info)
train_test_info.close()