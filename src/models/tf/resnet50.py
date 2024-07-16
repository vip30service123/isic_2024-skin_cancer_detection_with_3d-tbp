


from omegaconf import DictConfig, OmegaConf
import tensorflow as tf




class Resnet50:
    def __call__(self, config: DictConfig, show_summary: bool = False):
        model = tf.keras.applications.ResNet50(
                include_top=config['model']['include_top'],
                weights=config['model']['weights'],
                input_tensor=config['model']['input_tensor'],
                input_shape=config['model']['input_shape'],
                pooling=config['model']['pooling'],
                classes=config['model']['classes'],
                classifier_activation=config['model']['classifier_activation']
            )
        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        if show_summary:
            model.summary()

        return model
    



