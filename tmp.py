#!/usr/bin/env python
# coding=utf-8

# build lstm network

import numpy as np

import pandas
import math
import sys
import os
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from scipy.signal import argrelextrema

from lstm import *
from classifier import *
from burst_refinement import *
from prepare_v2 import *


def get_dates(filename):
    days = np.loadtxt(filename,
                      unpack=True, 
                      converters={0: mdates.strpdate2num('%Y%m%d')})
    # print days[:5]

    return days


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts, rst_file"
        exit(0)

    id_ = 1
    seq_len = 30
    epochs  = 100
    
    # load data
    print "load data"
    bursts_info, burst_series, non_burst_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    # use 1/3 as test set
    B_train, S_train, N_train, B_test, S_test, N_test = train_test_split(
             bursts_info, burst_series, non_burst_series)
    print "train/test size: ", len(S_train), len(S_test)

    t_data_c = get_samples_for_classfier_v2(S_train, N_train, seq_len) # [pos_samples, neg_samples]

    t_data_p_v = get_samples_for_predict_value(B_train, S_train, seq_len) # [seqs, values]
    t_data_p_t = get_samples_for_predict_period(B_train, S_train, seq_len) # [seqs, time_periods]
    t_data_e_v = get_samples_for_predict_end_value(B_train, S_train, seq_len) # [seqs,periods,st_values,peaks,end_values]

    t_data_r = get_samples_for_refine(sys.argv[4], sys.argv[5], sys.argv[6], id_) # [seqs]

    lstm_data = get_samples_for_lstm(S_train, seq_len)

    # build lstm model
    lstm_model = build_model(seq_len)
    lstm_model.fit(lstm_data[0], lstm_data[1], nb_epoch=epochs, batch_size=1, verbose=2)

    # build cluster model
    ap_model = build_refine_model(t_data_r)

    # if model already be trained and selected, use it!
    if os.path.exists("model/id" + str(id_) + "/clf.pkl") and \
       os.path.exists("model/id" + str(id_) + "/v_pdt.pkl") and \
       os.path.exists("model/id" + str(id_) + "/e_pdt.pkl") and \
       os.path.exists("model/id" + str(id_) + "/t_pdt.pkl") and \
       os.path.exists("model/id" + str(id_) + "/scores.npz"):
        clf = joblib.load('model/id' + str(id_) + '/clf.pkl') 
        v_pdt = joblib.load('model/id' + str(id_) + '/v_pdt.pkl')
        t_pdt = joblib.load('model/id' + str(id_) + '/t_pdt.pkl')
        e_pdt = joblib.load('model/id' + str(id_) + '/e_pdt.pkl')
        scores = np.load("model/id" + str(id_) + "/scores.npz")["scores"]

    else:
        # build clf model
        c_X, c_y = prepare_svm_input(t_data_c[0], t_data_c[1])
        clf1 = build_svm_model(c_X, c_y)
        clf2 = build_svm_model(c_X, c_y, "linear")

        # build peak value pred model
        v_X = prepare_prediction_input(t_data_p_v[0])
        v_pdt1 = build_prediction_model(v_X, t_data_p_v[1])
        # v_pdt2 = build_prediction_model(v_X, t_data_p_v[1], "svr")
        v_pdt2 = build_prediction_model(v_X, t_data_p_v[1], "linear")
        v_pdt3 = build_prediction_model(v_X, t_data_p_v[1], "bayes")

        # build period pred model
        t_X = prepare_period_prediction_input(t_data_p_t[0], t_data_p_t[1])
        t_pdt1 = build_prediction_model(t_X, t_data_p_t[2],)
        # t_pdt2 = build_prediction_model(t_X, t_data_p_t[2], "svr")
        t_pdt2 = build_prediction_model(t_X, t_data_p_t[2], "linear")
        t_pdt3 = build_prediction_model(t_X, t_data_p_t[2], "bayes")

        # build ed value pred model
        e_X = prepare_end_value_prediction_input(t_data_e_v[0], 
                                                 t_data_e_v[1], 
                                                 t_data_e_v[2], 
                                                 t_data_e_v[3])
        e_pdt1 = build_end_value_prediction_model(e_X, t_data_e_v[4])
        # e_pdt2 = build_end_value_prediction_model(e_X, t_data_e_v[4], "svr")
        e_pdt2 = build_end_value_prediction_model(e_X, t_data_e_v[4], "linear")
        e_pdt3 = build_end_value_prediction_model(e_X, t_data_e_v[4], "bayes")

        # test score
        # classifier score
        print "test classifier"
        dataset = get_samples_for_classfier_v2(S_test, N_test, seq_len)
        test_c_x, test_c_y = prepare_svm_input(dataset[0], dataset[1])
        score1 = []
        score1.append(clf1.score(test_c_x, test_c_y))
        score1.append(clf2.score(test_c_x, test_c_y))
        print "clf svc/linearsvc score: ", score1

        # peak value score
        test_v = get_samples_for_predict_value(B_test, S_test, seq_len)
        test_vx = prepare_prediction_input(test_v[0])
        score2 = []
        score2.append(mean_squared_error(v_pdt1.predict(test_vx), test_v[1]))
        # score2.append(mean_squared_error(v_pdt2.predict(test_vx), test_v[1]))
        score2.append(mean_squared_error(v_pdt2.predict(test_vx), test_v[1]))
        score2.append(mean_squared_error(v_pdt3.predict(test_vx), test_v[1]))

        # period score
        test_t = get_samples_for_predict_period(B_test, S_test, seq_len)
        test_tx = prepare_period_prediction_input(test_t[0], test_t[1])
        score3 = []
        score3.append(mean_squared_error(t_pdt1.predict(test_tx), test_t[2]))
        # score3.append(mean_squared_error(t_pdt2.predict(test_tx), test_t[2]))
        score3.append(mean_squared_error(t_pdt2.predict(test_tx), test_t[2]))
        score3.append(mean_squared_error(t_pdt3.predict(test_tx), test_t[2]))

        # end value score
        test_e = get_samples_for_predict_end_value(B_test, S_test, seq_len)
        test_ex = prepare_end_value_prediction_input(test_e[0], 
                                                     test_e[1], 
                                                     test_e[2], 
                                                     test_e[3])
        score4 = []
        score4.append(mean_squared_error(e_pdt1.predict(test_ex), test_e[4]))
        # score4.append(mean_squared_error(e_pdt2.predict(test_ex), test_e[4]))
        score4.append(mean_squared_error(e_pdt2.predict(test_ex), test_e[4]))
        score4.append(mean_squared_error(e_pdt3.predict(test_ex), test_e[4]))

        print "predict period, peak, end value score: ", score2, score3, score4    

        scores = [score1, score2, score3, score4]

        selected_ids = [np.argmax(score) for score in scores]
        clf = globals()['clf' + str(selected_ids[0] + 1)]
        v_pdt = globals()['v_pdt' + str(selected_ids[1] + 1)]
        t_pdt = globals()['t_pdt' + str(selected_ids[2] + 1)]
        e_pdt = globals()['e_pdt' + str(selected_ids[3] + 1)]

    # save models and scores
    joblib.dump(clf, 'model/id' + str(id_) + '/clf.pkl')
    joblib.dump(v_pdt, 'model/id' + str(id_) + '/v_pdt.pkl')
    joblib.dump(t_pdt, 'model/id' + str(id_) + '/t_pdt.pkl')
    joblib.dump(e_pdt, 'model/id' + str(id_) + '/e_pdt.pkl')
    np.savez('model/id' + str(id_) + '/scores.npz', scores=scores)

    # make predictions
    lstm_predictions = []
    our_predictions = []
    new_S = []
    msq_set = [] # mean square errors, (lstm_error, our_error) pair
    for series in S_test:
        ori_p = len(series)
        if ori_p <= seq_len:
            continue
        seq = series[0:seq_len]
        lstm_pred = np.empty_like(series[seq_len:])
        
        # lstm recurrent prediction
        for i in range(len(series)-seq_len):
            lstm_pred[i] = lstm_model.predict(np.reshape(seq,(1,1,seq_len)))
            seq[0:seq_len-1] = seq[1:seq_len]
            seq[-1] = lstm_pred[i]

        # our prediction
        seq = series[0:seq_len] # reset seq
        peak = predict_burst_value(v_pdt, seq)
        print "peak, pred_peak: ", np.amax(series), peak
        period = predict_burst_period(t_pdt, peak, seq)
        print "period, pred_period: ", ori_p, period
        ed_value = predict_end_value(e_pdt, seq, period, peak)
        print "ed_value, pred_ed_value: ", series[-1], ed_value
        new_seq, st, ed, last_p, start = reshape_orginal_seq(seq, period)

        reshaped_rst = knn_prediction(ap_model, t_data_r, new_seq, period, peak, ed_value, st, ed)
        if reshaped_rst is None:
            continue
        new_burst = stretch_burst(reshaped_rst[0], period, seq[0])

        if period > ori_p:
            our_pred = new_burst[seq_len:ori_p]
        else:
            our_pred = np.zeros(ori_p-seq_len)
            our_pred[:period-seq_len] = new_burst[seq_len:]

        msq = np.zeros(2)
        msq[0] = mean_squared_error(series[seq_len:], lstm_pred)
        msq[1] = mean_squared_error(series[seq_len:], our_pred)
        print "mean_squared_error: ", msq
        msq_set.append(msq)
        lstm_predictions.append(lstm_pred)
        our_predictions.append(our_pred)
        new_S.append(series)


    # invert predictions
    # dataset, new_pre = inverse_data(scaler, [dataset, [new_pre]])
    print "overall msq: ", np.mean(np.array(msq_set), axis=0)

    np.savez(sys.argv[7], our_pred=our_predictions, 
                          pred=lstm_predictions,
                          msq=msq_set,
                          series=new_S,
                          scores=scores,
                          seq_len=seq_len)
