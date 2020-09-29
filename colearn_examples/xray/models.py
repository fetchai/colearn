from tensorflow.compat import v1 as tf

from colearn_examples.keras_learner import KerasLearner


tf.disable_v2_behavior()


class XraySuperminiLearner(KerasLearner):
    def _get_model(self):
        # Minimalistic model
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.MaxPooling2D((4, 4), name="pool1")(x)
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.GlobalMaxPool2D()(x)

        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayResnet50Learner(KerasLearner):
    def _get_model(self):
        # Resnet50
        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(self.config.width, self.config.height, 1),
        )

        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = resnet(input_img)
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayPretrainedResnet50Learner(KerasLearner):
    def _get_model(self):
        # Resnet50
        resnet = tf.keras.applications.ResNet50(include_top=False)

        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(3, (1, 1), activation="linear", padding="same")(
            input_img
        )
        x = resnet(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayDropout2Learner(KerasLearner):
    def _get_model(self):
        # Minimalistic model
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        x = tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="Conv3_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn3")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

        x = tf.keras.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="Conv4_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn4")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool4")(x)

        x = tf.keras.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="Conv5_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn5")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool5")(x)

        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(256, activation="relu", name="fc1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn6")(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Dense(128, activation="relu", name="fc2")(x)
        x = tf.keras.layers.BatchNormalization(name="bn7")(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Dense(64, activation="relu", name="fc3")(x)
        x = tf.keras.layers.BatchNormalization(name="bn8")(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc4"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayDropoutLearner(KerasLearner):
    def _get_model(self):
        # Minimalistic model
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.MaxPooling2D((4, 4), name="pool1")(x)
        x = tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.MaxPooling2D((4, 4), name="pool2")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(128, activation="relu", name="fc1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn3")(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Dense(64, activation="relu", name="fc2")(x)
        x = tf.keras.layers.BatchNormalization(name="bn4")(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc3"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayMiniLearner(KerasLearner):
    def _get_model(self):
        # Minimalistic model
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.MaxPooling2D((4, 4), name="pool1")(x)
        x = tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.MaxPooling2D((4, 4), name="pool2")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayOneMBLeaner(KerasLearner):
    def _get_model(self):
        # Minimalistic model - 1MB
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1_1")(x)
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv1_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn1_2")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2_1")(x)
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv2_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2_2")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv3_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn3_1")(x)
        x = tf.keras.layers.SeparableConv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv3_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn3_2")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

        x = tf.keras.layers.SeparableConv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv4_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn4_1")(x)
        x = tf.keras.layers.SeparableConv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv4_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn4_2")(x)
        x = tf.keras.layers.AvgPool2D((4, 4), name="pool4")(x)

        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayPretrainedResNet50Learner(KerasLearner):
    def _get_model(self):
        resnet50_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False, input_shape=(self.config.width, self.config.height, 3)
        )

        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )

        # Preprocess input for pretrained RESNET50
        x = tf.keras.layers.Concatenate(axis=-1)([input_img, input_img, input_img])
        x = x * 255
        x = tf.keras.applications.resnet50.preprocess_input(x)

        x = resnet50_model(x)
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)

        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class XrayVGG16Learner(KerasLearner):
    def _get_model(self):
        vgg16_model = tf.keras.applications.vgg16.VGG16(
            include_top=False, input_shape=(self.config.width, self.config.height, 3)
        )

        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )

        # Preprocess input for pretrained VGG16
        x = tf.keras.layers.Concatenate(axis=-1)([input_img, input_img, input_img])
        x = x * 255
        x = tf.keras.applications.vgg16.preprocess_input(x)

        x = vgg16_model(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)

        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model
