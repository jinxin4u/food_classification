import keras

class AccuracyHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append( logs.get('acc') )