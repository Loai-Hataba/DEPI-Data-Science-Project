import tensorflow as tf
from keras.models import load_model as load_keras_model

def adapt_keras_model(model_path, new_input_shape):
    """تعديل نموذج Keras ليتقبل شكل إدخال جديد"""
    original_model = load_keras_model(model_path)
    
    # إنشاء نموذج جديد مع طبقة إدخال معدلة
    new_input = tf.keras.layers.Input(shape=new_input_shape)
    
    # تجميع الطبقات مع تعديل الأوزان
    x = new_input
    for layer in original_model.layers[1:]:  # تخطي طبقة الإدخال الأصلية
        if isinstance(layer, tf.keras.layers.Dense):
            # إنشاء طبقة جديدة مع عدد الوحدات المناسب
            new_layer = tf.keras.layers.Dense(
                units=layer.units,
                activation=layer.activation,
                kernel_initializer=tf.constant_initializer(layer.get_weights()[0][:new_input_shape[0], :]),
                bias_initializer=tf.constant_initializer(layer.get_weights()[1])
            )
            x = new_layer(x)
        else:
            x = layer(x)
    
    return tf.keras.Model(inputs=new_input, outputs=x)