import tensorflow as tf

def compile_model(architecture_string, input_shape):
    layers = []
    arch_parts = architecture_string.split('->')
    for i, part in enumerate(arch_parts):
        part = part.strip()
        if part.startswith('Dense'):
            params = part[6:-1]  # Remove 'Dense(' and ')'
            try:
                units, activation = params.split(',')
                units = int(units.strip())
                activation = activation.strip()
            except Exception:
                raise ValueError(f"Error en la sintaxis de la capa: {part}")
            if i == 0:
                layers.append(tf.keras.layers.Dense(units, activation=activation, input_shape=input_shape))
            else:
                layers.append(tf.keras.layers.Dense(units, activation=activation))
        else:
            raise ValueError(f"Tipo de capa no soportado: {part}")
    model = tf.keras.Sequential(layers)
    return model
