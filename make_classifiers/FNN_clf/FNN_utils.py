import tensorflow.keras.backend as K
import numpy as np
import os

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

# class FNN_model():
#     def __init__(self, param, DB):
#         self.DB = DB
#         self.nb_features = DB.drugs.nb + DB.proteins.nb
#         # here the metric to minimize the error is the accuracy, but it should
#         # be the precision
#         self.dataset = Dataset(name='DrugBank',
#                                n_outputs=1,
#                                loss='binary_crossentropy', 
#                                final_activation='sigmoid',
#                                metrics=[keras.metrics.binary_accuracy], 
#                                ea_metric='aupr', 
#                                ea_mode='max')
#         self.n_epochs = param['n_epochs']
#         self.init_lr = param['init_lr']
#         self.layers_units = param['layers_units']
#         self.n_layers = len(self.layers_units)
#         self.BN = param['BN']
#         if param['reg'] == 0:
#             self.kreg, self.breg = None, None
#         else:
#             self.kreg, self.breg = regularizers.l2(param['reg']), regularizers.l2(param['reg'])
#         self.drop = param['drop']
#         self.patience_early_stopping = param['patience']
#         self.lr_scheduler = param['lr_scheduler']
#         self.queue_size_for_generator = 1000
#         self.cpu_workers_for_generator = 5

def FNN_save_model(model, filename):
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])

    for name, w in layer_dict.items():
        np.save(filename + name.replace('/', '_'), K.get_value(w))
        print(name, 'SAVED')
    pass

def FNN_load_model(model, filename):
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])
    for name, w in layer_dict.items():
        print(name)
        if os.path.isfile(filename + name.replace('/', '_') + '.npy'):
            print('FOUND', filename + name.replace('/', '_') + '.npy')
            value = np.load(filename + name.replace('/', '_') + '.npy')
            K.set_value(w, value)
        else:
            print('NOT FOUND', filename + name.replace('/', '_') + '.npy')
            # layer_dict['some_name'].set_weights(w)
    print('done load_FFN_weights')
    return model