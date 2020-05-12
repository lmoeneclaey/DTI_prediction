from sklearn import metrics
import keras.utils
import numpy as np
# from keras.callbacks import Callback
#from src.utils.DB_utils import MISSING_DATA_MASK_VALUE

MISSING_DATA_MASK_VALUE = -1


def sround(score):
    return round(score * 100, 2)


def get_clf_perf(y_pred, y_true,
                 perf_names=['AUPR', 'ROCAUC', 'F1', 'Recall', "Precision", "ACC"]):
    dict_perf_fct = {'AUPR': aupr, 'ROCAUC': rocauc, 'F1': f1,
                     'Recall': recall, "Precision": precision, "ACC": acc}
    dict_perf = {}
    # import pdb; pdb.Pdb().set_trace()
    for perf_name in perf_names:
        if perf_name in ['F1', 'Recall', "Precision", "ACC"]:
            y_pred_ = y_pred.copy()
            if len(y_pred_.shape) > 1 and y_pred_.shape[1] != 1:
                y_pred_ = (y_pred_ == y_pred_.max(axis=1)[:, None]).astype(int)
            else:
                y_pred_ = np.round(y_pred_)
            # import pdb; pdb.Pdb().set_trace()
        else:
            y_pred_ = y_pred
        dict_perf[perf_name] = dict_perf_fct[perf_name](y_true, y_pred_)
    return dict_perf


def get_regr_perf(y_pred, y_true,
                  perf_names=['MSE']):
    dict_perf_fct = {'MSE': mse}
    dict_perf = {}
    for perf_name in perf_names:
        dict_perf[perf_name] = dict_perf_fct[perf_name](y_true, y_pred)

    return dict_perf


def del_missing_data(y_true, y_pred):
    list_y_true, list_y_pred, missing_ratio = [], [], []
    # print(y_true)
    # print(y_true.shape)
    if len(y_pred.shape) > 1:
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]
    if len(y_pred.shape) > 1:
        if len(y_true.shape) == 1:
            y_true = keras.utils.to_categorical(y_true)
        for i_out in range(y_pred.shape[1]):
            ind_missing = np.where(y_true[:, i_out] == MISSING_DATA_MASK_VALUE)[0]
            list_y_true.append(np.delete(y_true[:, i_out], ind_missing))
            list_y_pred.append(np.delete(y_pred[:, i_out], ind_missing))
            missing_ratio.append(float(y_true.shape[0] - len(ind_missing)))
        missing_ratio = np.array(missing_ratio) / np.sum(missing_ratio)
    else:
        ind_missing = np.where(y_true[:] == MISSING_DATA_MASK_VALUE)[0]
        list_y_true.append(np.delete(y_true[:], ind_missing))
        list_y_pred.append(np.delete(y_pred[:], ind_missing))
        missing_ratio = np.array([1.])
    # import pdb; pdb.Pdb().set_trace()
    return list_y_true, list_y_pred, missing_ratio


def mcc(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        mcc = []
        for i_out in range(len(list_y_pred)):
            # import pdb; pdb.Pdb().set_trace()
            mcc.append(round(metrics.matthews_corrcoef(list_y_true[i_out], list_y_pred[i_out]), 4))
        return (round(np.dot(mcc, ratio_per_output), 4), mcc)
    else:
        mcc = round(metrics.matthews_corrcoef(y_true, y_pred), 4)
        return (mcc, [mcc])


def acc(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        acc = []
        for i_out in range(len(list_y_pred)):
            # import pdb; pdb.Pdb().set_trace()
            acc.append(round(metrics.accuracy_score(list_y_true[i_out], list_y_pred[i_out]), 4))
        return (round(np.dot(acc, ratio_per_output), 4), acc)
    else:
        acc = round(metrics.accuracy_score(y_true, y_pred), 4)
        return (acc, [acc])


def aupr(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        # import pdb; pdb.Pdb().set_trace()
        aupr = []
        for i_out in range(len(list_y_pred)):
            aupr.append(sround(
                metrics.average_precision_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(aupr, ratio_per_output), 2), aupr)
    else:
        aupr = sround(metrics.average_precision_score(y_true, y_pred))
        return (aupr, [aupr])


def rocauc(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        auc = []
        for i_out in range(len(list_y_pred)):
            auc.append(sround(
                metrics.roc_auc_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(auc, ratio_per_output), 2), auc)
    else:
        rocauc = sround(metrics.roc_auc_score(y_true, y_pred))
        return (rocauc, [rocauc])


def f1(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        f1 = []
        for i_out in range(len(list_y_pred)):
            f1.append(sround(
                metrics.f1_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(f1, ratio_per_output), 2), f1)
    else:
        f1 = sround(metrics.f1_score(y_true, y_pred))
        return (f1, [f1])


def recall(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        recall = []
        for i_out in range(len(list_y_pred)):
            recall.append(sround(
                metrics.recall_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(recall, ratio_per_output), 2), recall)
    else:
        recall = sround(metrics.recall_score(y_true, y_pred))
        return (recall, [recall])


def precision(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        recall = []
        for i_out in range(len(list_y_pred)):
            recall.append(sround(
                metrics.precision_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(recall, ratio_per_output), 2), recall)
    else:
        precision = sround(metrics.precision_score(y_true, y_pred))
        return (precision, [precision])


def mse(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        mse = []
        for i_out in range(len(list_y_pred)):
            mse.append(round(
                metrics.mean_squared_error(list_y_true[i_out], list_y_pred[i_out]), 5))
        return (round(np.dot(mse, ratio_per_output), 5), mse)
    else:
        mse = round(metrics.mean_squared_error(y_true, y_pred), 5)
        return (mse, [mse])
