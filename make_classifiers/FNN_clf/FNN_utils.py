import tensorflow.keras.backend as K
import numpy as np
import os

from DTI_prediction.utils.package_utils import get_lr_metric, get_imbalance_data, OnAllValDataEarlyStopping

from sklearn import metrics

from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, LearningRateScheduler

# from keras import backend as K
# from keras import regularizers
# import keras

class Dataset():
    def __init__(self, name, n_outputs, loss, final_activation, metrics, ea_metric, ea_mode):
        self.name = name
        self.n_outputs = n_outputs
        self.loss = loss
        self.final_activation = final_activation
        self.metrics = metrics
        self.ea_metric = ea_metric  # earlystopping_metric
        self.ea_mode = ea_mode
        # earlystopping_mode : in min mode, training will stop when the
        # quantity monitored has stopped decreasing; in max mode
        # it will stop when the quantity monitored has stopped increasing;
        # in auto mode, the direction is automatically inferred from the
        # name of the monitored quantity.

class FNN_model():
    
    def __init__(self, param, DB, X_prot, X_mol):
        self.DB = DB
        self.nb_features = X_prot.shape[1] + X_mol.shape[1]
        # here the metric to minimize the error is the accuracy, but it should
        # be the precision
        self.dataset = Dataset(name='DrugBank', 
                               n_outputs=1,
                               loss='binary_crossentropy', 
                               final_activation='sigmoid',
                               metrics=[binary_accuracy], 
                               ea_metric='aupr', 
                               ea_mode='max')
        self.n_epochs = param['n_epochs']
        # lr : learning rate
        self.init_lr = param['init_lr']
        self.layers_units = param['layers_units']
        self.n_layers = len(self.layers_units)
        self.BN = param['BN']
        if param['reg'] == 0:
            self.kreg, self.breg = None, None
        else:
            self.kreg, self.breg = regularizers.l2(param['reg']), regularizers.l2(param['reg'])
        self.drop = param['drop']
        self.patience_early_stopping = param['patience']
        self.lr_scheduler = param['lr_scheduler']
        # self.queue_size_for_generator = 1000
        # self.cpu_workers_for_generator = 5
        
    # build an FFN
    def build(self):
        
        # Input of the Neural Network     
        batch = Input(shape=(self.nb_features,), name='input')
        x = batch  # BatchNormalization(axis=-1)(batch)
        for nl in range(self.n_layers):
            # Fully feed-forward network hidden layer "Dense()"
            # activation "relu"
            x = Dense(self.layers_units[nl], 
                      activation='relu',
                      # weights initialisation Xavier Glorot
                      kernel_initializer='glorot_uniform', 
                      # biais initiatilisation : zeros
                      bias_initializer='zeros',
                      kernel_regularizer=self.kreg, 
                      bias_regularizer=self.breg,
                      activity_regularizer=None, 
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name='layer_' + str(nl))(x)
            # Drop out
            if self.drop != 0.:
                x = Dropout(self.drop, 
                            noise_shape=None, 
                            seed=None)(x)
            # Batch Normalization
            if self.BN is True:
                x = BatchNormalization(axis=-1)(x)  # emb must be (batch_size, emb_size)
        # Output layer
        # final activation: sigmoid
        predictions = Dense(self.dataset.n_outputs,
                            activation=self.dataset.final_activation, 
                            name='output')(x)
        
        optimizer = optimizers.Adam(lr=self.init_lr)
        lr_metric = get_lr_metric(optimizer)
        model = Model(inputs=[batch], 
                      outputs=[predictions])
        model.compile(optimizer=optimizer,
                      loss={'output': self.dataset.loss},
                      loss_weights={'output': 1.},
                      metrics={'output': self.dataset.metrics + [lr_metric]})
        self.model = model
        
    def fit(self, X_tr, y_tr, X_val, y_val):
        callbacks = self.get_callbacks('fea', X_val, y_val)
        # balance predictor in case of uneven class distribution in train data
        # self.dataset.name should be useless
        inverse_class_proportion = get_imbalance_data(y_tr, self.dataset.name)
        if inverse_class_proportion is not None:
            print('WARNING: imbalance class proportion ', inverse_class_proportion)
        # import pdb; pdb.Pdb().set_trace()

        self.model.fit(X_tr, 
                       np.array(y_tr), 
                       steps_per_epoch=None, 
                       epochs=self.n_epochs,
                       verbose=1, 
                       callbacks=callbacks, 
                       validation_data=(X_val, np.array(y_val)),
                       validation_steps=None, 
                       class_weight=inverse_class_proportion,
                       shuffle=True, 
                       initial_epoch=0)
    
    # a mon avis il y a une erreur sur gen_bool, val_gen, etc ...
    def get_callbacks(self, gen_bool, val_gen, val_y, train_gen=None, train_y=None):
        # callbacks when fitting
        early_stopping = OnAllValDataEarlyStopping(self.dataset.name, 
                                                   gen_bool, 
                                                   val_gen, 
                                                   val_y, 
                                                   train_gen, 
                                                   train_y,
                                                   qsize=1000, 
                                                   workers=5,
                                                   monitor=self.dataset.ea_metric, 
                                                   mode=self.dataset.ea_mode,
                                                   # minimum change to qualify as an improvement
                                                   min_delta=0,  
                                                   patience=self.patience_early_stopping, 
                                                   verbose=1, 
                                                   restore_best_weights=True)
        csv_logger = CSVLogger('temp.log')  # streams epoch results to a csv
        if self.lr_scheduler['name'] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', 
                                             factor=0.2, 
                                             patience=2,
                                             verbose=1, 
                                             mode='auto',
                                             min_delta=0.0001, 
                                             min_lr=0.001)
        elif self.lr_scheduler['name'] == 'LearningRateScheduler':
            rate = self.lr_scheduler['rate']
            # reducing the learning rate by "rate" every 2 epochs
            lr_scheduler = LearningRateScheduler(
                lambda epoch: self.init_lr * rate ** (epoch // 2), verbose=0)
        callbacks = [early_stopping, lr_scheduler] + [csv_logger]
        return callbacks
    
    def def_pred_model(self):
        return Model(inputs=self.model.input, 
                     outputs=self.model.get_layer('output').output)

    def predict(self, X_te):
        pred_model = self.def_pred_model()
        return pred_model.predict(X_te)
    
    def trainable_weights(self):
        return self.model.trainable_weights 
