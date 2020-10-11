#@title Import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Load the datasets from the internet.
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Scale the label values.
scale_factor = 1000.0
# Scale the training set's label.
train_df["median_house_value"] /= scale_factor
# Scale the test set's label.
test_df["median_house_value"] /= scale_factor


# @title Define the functions build_model and train_model that build and train a model
# build_model defines the model's topography
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


# trains the model, outputting not only the loss value for the training set but also the loss value for the validation set.
def train_model(model, df, feature, label, my_epochs,
                my_batch_size=None, my_validation_split=0.1):
    """Feed a dataset into the model in order to train it."""

    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=my_batch_size,
                        epochs=my_epochs,
                        validation_split=my_validation_split)

    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch

    # Isolate the root mean squared error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse, history.history


print("Defined the build_model and train_model functions.")


# @title Define the plotting function
# The `plot_the_loss_curve` function plots loss vs. epochs for both the training set and the validation set.
def plot_the_loss_curve(epochs, mae_training, mae_validation):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()

    # We're not going to plot the first epoch, since the loss on the first epoch
    # is often substantially greater than the loss for other epochs.
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()


print("Defined the plot_the_loss_curve function.")

# The following variables are the hyperparameters.
learning_rate = 0.08
epochs = 30
batch_size = 100

# Split the original training set into a reduced training set and a
# validation set.
validation_split = 0.2
# The original training set contains 17.000 examples.
# Therefore, a validation_split od 0.2 means
# 17,000 * 0.2 ~= 3,400 examples will become the validation set.
# 17,000 * 0.8 ~= 13,600 examples will become the new training set.

# Identify the feature and the label.
my_feature = "median_income"  # the median income on a specific city block.
my_label = "median_house_value" # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based
# solely on the neighborhood's median income.

# Discard any pre-existing version of the model.
my_model = None

# Invoke the functions to build and train the model.
my_model = build_model(learning_rate)
epochs, rmse, history = train_model(my_model, train_df, my_feature,
                                    my_label, epochs, batch_size,
                                    validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                    history["val_root_mean_squared_error"])

# Evidently, the data in the training set isn't similar enough to the data in the validation set.
# As with most issues in machine learning, the problem is rooted in the data itself.
# To solve this mystery of why the training set and validation set aren't almost identical
# here you have an example.

# Examine examples 0 through 4 and examples 25 through 29
# of the training set
print(train_df.head(n=1000))

# The original training set is sorted by longitude.
# Apparently, longitude influences the relationship of
# total_rooms to median_house_value.


# FIX THIS PROBLEM!
# The following variables are the hyperparameters.
learning_rate = 0.08
epochs = 70
batch_size = 100

# Split the original training set into a reduced training set and a
# validation set.
validation_split = 0.2

# Identify the feature and the label.
my_feature = "median_income"  # the median income on a specific city block.
my_label = "median_house_value" # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based
# solely on the neighborhood's median income.

# Discard any pre-existing version of the model.
my_model = None

#Shuffle the data in the training set by adding the following line anywhere before you call train_model
shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

# Invoke the functions to build and train the model. Train on the shuffled
# training set.
my_model = build_model(learning_rate)

# Pass shuffled_train_df (instead of train_df) as the second argument
# to train_model so that the call becomes as follows:
epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature,
                                    my_label, epochs, batch_size,
                                    validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                    history["val_root_mean_squared_error"])

# Examine examples 0 through 4 and examples 25 through 29
# of the training set and compare to train_df.
print(shuffled_train_df.head(n=1000))

x_test = test_df[my_feature]
y_test = test_df[my_label]

# The test set usually acts as the ultimate judge of a model's quality.
# The test set can serve as an impartial judge because its examples haven't been used in training the model.
# Run the following code cell to evaluate the model with the test set:
results = my_model.evaluate(x_test, y_test, batch_size=batch_size)