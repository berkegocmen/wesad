MODEL_OUTPUT = '../models/'

TRAINING_FILE = '../data/S_train.csv'

VALIDATION_FILE = '../data/S_valid.csv'

TEST_FILE = '../data/S_test.csv'

training_files = {'tf_1': '../data/S_csvs/S14.csv',
                  'tf_2': '../data/S_csvs/S15.csv',
                  'tf_3': '../data/S_csvs/S4.csv',
                  'tf_4': '../data/S_csvs/S5.csv',
                  'tf_5': '../data/S_csvs/S6.csv',
                  'tf_6': '../data/S_csvs/S7.csv',
                  'tf_7': '../data/S_csvs/S8.csv',
                  'tf_8': '../data/S_csvs/S9.csv',
                  'tf_9': '../data/S_csvs/S10.csv',
                  'tf_10': '../data/S_csvs/S11.csv',
                  'tf_11': '../data/S_csvs/S13.csv',
                  'tf_12': '../data/S_csvs/S16.csv',
                  'tf_13': '../data/S_csvs/S17.csv',
                  }

# validation_files = {'vf_1': '../data/S_csvs/S16.csv',
#                     'vf_2': '../data/S_csvs/S17.csv'}

test_files = {'f_1': '../data/S_csvs/S2.csv',
              'f_2': '../data/S_csvs/S3.csv'}

PARAMS = {'max_bin': 9, 'max_depth': 87, 'lambda_l1': 1.1799718495758467, 'lambda_l2': 47.56543205622963,
          'num_leaves': 112, 'feature_fraction': 0.511093310643247, 'bagging_fraction': 0.9761885828245985,
          'bagging_freq': 10, 'min_child_samples': 10, 'learning_rate': 0.09363262777446904}

THREE_CLASS_FOLDS = '../data/3class/train_folds.csv'
THREE_CLASS_TEST = '../data/3class/test.csv'

THREE_CLASS_FOLDS_v2 = '../data/3class/train_v2.csv'
THREE_CLASS_TEST_v2 = '../data/3class/test_v2.csv'

THREE_CLASS_EXTRACTED_FOLDS = '../data/3class/folds_extracted.csv'
THREE_CLASS_EXTRACTED_TEST = '../data/3class/test_extracted.csv'

THREE_CLASS_EXTRACTED_FOLDS_v2 = '../data/3class/folds_extracted_v2.csv'
THREE_CLASS_EXTRACTED_TEST_v2 = '../data/3class/test_extracted_v2.csv'
