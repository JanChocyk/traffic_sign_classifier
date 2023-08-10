from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50

class ModelVGGNetSmall():
    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation

    def build(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units=1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.num_classes, activation=self.final_activation))

        return model

class ModelResNet50():
    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation

    def build(self):
        '''
        Requirement input shape (224, 224, 3).
        '''
        base_model = ResNet50(input_shape=self.input_shape, include_top=False, weights=None)
        flattened = Flatten()(base_model.output)
        output_layer = Dense(units=self.num_classes, activation=self.final_activation)(flattened)
        model = Model(inputs=base_model.input, outputs=output_layer)

        return model
