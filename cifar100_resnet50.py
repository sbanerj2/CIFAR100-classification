from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from load_cifar100 import load_cifar100_data
from keras import backend as K

#K.set_image_dim_ordering('tf')

def resnet50_model(num_classes):
	
	
	# create the base pre-trained model
	base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

	# add a global spatial average pooling layer
	x = base_model.output
	print("Shape:", x.shape)
	x_newfc = x = GlobalAveragePooling2D()(x)

	# and a logistic layer -- let's say we have 100 classes
	x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

	# this is the model we will train
	# Create another model with our customized softmax
	model = Model(inputs=base_model.input, outputs=x_newfc)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional resnet50 layers
	for layer in base_model.layers:
		layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	# Learning rate is changed to 0.001
	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',  'top_k_categorical_accuracy'])
	return model
  

if __name__ == '__main__':

    # Example to fine-tune on samples from Cifar100

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 100 
    batch_size = 16 
    nb_epoch = 50

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar100_data(img_rows, img_cols)

    # Load our model
    model = resnet50_model(num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    ## save model
    model_json = model.to_json()
    with open("cifar100_resnet50.json", "w") as json_file:
    	json_file.write(model_json)
    model.save_weights("cifar100_resnet50.h5")
    print("Saved model to disk")
    
    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    scores = log_loss(Y_valid, predictions_valid)
    print("Cross-entropy loss score",scores)
    
    ## evaluate modelon test data:
    score = model.evaluate(X_valid, Y_valid, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

