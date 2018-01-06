from keras.layers import Input, Embedding, Dense, Reshape, concatenate
from keras.models import Model

def trainModel(BoWSize, EncodedSize, input_dim, output_dim, inputs, output, epochs, batch_size, outfile_name):
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    user_input_embed = Embedding(output_dim=output_dim, input_dim=input_dim, input_length=1)(user_input)
    user_input_embed = Reshape((output_dim,))(user_input_embed)
    item_input = Input(shape=(BoWSize,), name='item_input')

    # auto-encoder part
    ae01 = Dense(int(BoWSize), activation='relu')(item_input)
    ae02 = Dense(int(BoWSize/2), activation='relu')(ae01)
    ae03 = Dense(int(BoWSize/4), activation='relu')(ae02)
    ae_bottleneck = Dense(EncodedSize, activation='relu')(ae03)
    ae05 = Dense(int(BoWSize/4), activation='relu')(ae_bottleneck)
    ae06 = Dense(int(BoWSize/2), activation='relu')(ae05)
    ae07 = Dense(int(BoWSize), activation='relu')(ae06)
    ae08 = Dense(int(BoWSize), activation='relu')(ae07)

    merge = concatenate([user_input_embed, ae_bottleneck])
    # hidden layers
    x = Dense(2000, activation='relu')(merge)
    x = Dense(500, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(30, activation='relu')(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    encoder_output = Dense(BoWSize, activation='sigmoid', name='encoder_output')(ae08)

    # define model
    model = Model(inputs=[user_input, item_input], outputs=[main_output, encoder_output])
    model.compile(optimizer='adam',
              loss={'main_output': 'binary_crossentropy', 'encoder_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'encoder_output': 0.3})

    # train & save model
    model.fit([inputs[0], inputs[1]], [output, inputs[1]], epochs=epochs, batch_size=batch_size)
    model.save(outfile_name)
    del model


