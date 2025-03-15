import matplotlib.pyplot as plt
import tensorflow as tf


def random_flip(input_image, input_mask):
    """
    random flip of input image and mask
    """
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask


def normalize(input_image, input_mask):
    """
    normalize input image pixel values to be from [0, 1]
    subtract 1 from the mask labels to have a range of [0, 2]
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    """
    resize, normalize and flip the training data
    """
    input_image = tf.image.resize(datapoint["image"], (128, 128), method="nearest")
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128), method="nearest")
    input_image, input_mask = random_flip(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    """
    resize and normalized the test data
    """
    input_image = tf.image.resize(datapoint["image"], (128, 128), method="nearest")
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128), method="nearest")
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# Visualization

class_names = ["pet", "background", "outline"]


def display_with_metrics(display_list, iou_list, dice_score_list):
    """
    display a list of images/masks and overlays a list of IOU and dice score
    """

    metrics_by_id = [
        (idx, iou, dice_score)
        for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list))
        if iou > 0.0
    ]
    metrics_by_id.sort(key=lambda x: x[1], reverse=True)

    display_string_list = [
        f"{class_names[idx]} IOU: {iou} Dice Score: {dice_score}"
        for idx, iou, dice_score in metrics_by_id
    ]
    display_string = "\n\n".join(display_string_list)

    display(display_list, ["Image", "Predicted Mask", "True Mask"], display_string)


def display(display_list, titles=[], display_string=None):
    """
    display a list of images/masks
    """
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        if display_string and i == 1:
            plt.xlabel(display_string, fontsize=12)
        img_arr = tf.keras.preprocessing.image.array_to_img(display_list[i])
        plt.imshow(img_arr)

    plt.show()




def show_image_from_dataset(dataset):
    """
    display the first image and its mask from a dataset
    """

    for image, mask in dataset.take(1):
        sample_image = image
        sample_mask = mask

    display([sample_image, sample_mask], ["Image", "True Mask"])


def plot_metrics(metric_name, history, title, ylim=5):
    """
    plot a given metric from the model history
    """

    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color="blue", label=metric_name)
    plt.plot(history.history["val_" + metric_name], color="green", label=metric_name)
