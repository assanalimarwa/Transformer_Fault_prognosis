def mhsa_bigru(input_shape: Tuple[int, ...], bigru_units: int, num_heads: int, head_dim: int, num_layers: int) -> model.Model:
    """
    Constructs a model with stacked BiGRU layers followed by a Multi-Head Self-Attention (MHSA) layer.

    Args:
        input_shape (Tuple[int, ...]): Shape of the input data (e.g., (60, 1)).
        bigru_units (int): Number of units in each BiGRU layer. Default is 32.
        num_heads (int): Number of attention heads for the MHSA layer. Default is 4.
        num_layers (int): Number of stacked BiGRU layers. Default is 5.

    Returns:
        models.Model: A compiled Keras model.
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Stack BiGRU layers dynamically based on num_layers
    x = inputs
    for _ in range(num_layers):
        x = layers.Bidirectional(layers.GRU(bigru_units, return_sequences=True))(x)

    # Multi-Head Self-Attention layer
    mhsa = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x, x)

    # Flatten the MHSA output
    flatten = layers.Flatten()(mhsa)

    # Output layer (single neuron for regression)
    outputs = layers.Dense(1)(flatten)

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model