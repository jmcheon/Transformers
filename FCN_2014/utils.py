import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

# pixel labels in the video frames
class_names = [
    "sky",
    "building",
    "column/pole",
    "road",
    "side walk",
    "vegetation",
    "traffic light",
    "fence",
    "vehicle",
    "pedestrian",
    "byciclist",
    "void",
]

# generate a list that contains one color for each class
colors = sns.color_palette(None, len(class_names))


def fuse_with_pil(images):
    """
    Creates a blank image and pastes input images

    Args:
        images (List[ndarray]): ndarray representations of the images to paste

    Returns:
        PIL Image object containing the images
    """

    widths = (image.shape[1] for image in images)
    heights = (image.shape[0] for image in images)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        pil_image = Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset, 0))
        x_offset += im.shape[1]

    return new_im


def give_color_to_annotation(annotation):
    """
    Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
    the third axis represents the color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation

    Args:
        annotation (ndarray): label map array

    Returns:
        the annotation array with an additional color channel/axis
    """
    seg_img = np.zeros((annotation.shape[0], annotation.shape[1], 3)).astype("float")

    for c in range(12):
        segc = annotation == c
        seg_img[:, :, 0] += segc * (colors[c][0] * 255.0)
        seg_img[:, :, 1] += segc * (colors[c][1] * 255.0)
        seg_img[:, :, 2] += segc * (colors[c][2] * 255.0)

    return seg_img


def show_predictions(image, labelmaps, titles, iou_list, dice_score_list):
    """
    Displays the images with the ground truth and predicted label maps

    Args:
        image (ndarray): the input image
        labelmaps (list of arrays): contains the predicted and ground truth label maps
        titles (list of str): display headings for the images to be displayed
        iou_list (list of floats): the IOU values for each class
        dice_score_list (list of floats): the Dice Score for each vlass
    """

    true_img = give_color_to_annotation(labelmaps[1])
    pred_img = give_color_to_annotation(labelmaps[0])

    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_img, true_img])

    metrics_by_id = [
        (idx, iou, dice_score)
        for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list))
        if iou > 0.0
    ]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

    display_string_list = [
        "{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score)
        for idx, iou, dice_score in metrics_by_id
    ]
    display_string = "\n\n".join(display_string_list)

    plt.figure(figsize=(15, 4))

    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx + 1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)


def show_annotation_and_image(image, annotation):
    """
    Displays the image and its annotation side by side

    Args:
        image (ndarray): the input image
        annotation (ndarray): the label map
    """
    new_ann = np.argmax(annotation, axis=2)
    seg_img = give_color_to_annotation(new_ann)

    image = image + 1
    image = image * 127.5
    image = np.uint8(image)
    images = [image, seg_img]

    images = [image, seg_img]
    fused_img = fuse_with_pil(images)
    plt.imshow(fused_img)


def list_show_annotation(dataset):
    """
    Displays images and its annotations side by side

    Args:
        dataset (tf Dataset): batch of images and annotations
    """

    ds = dataset.unbatch()
    ds = ds.shuffle(buffer_size=100)

    plt.figure(figsize=(25, 15))
    plt.title("Images And Annotations")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

    # we set the number of image-annotation pairs to 9
    for idx, (image, annotation) in enumerate(ds.take(9)):
        plt.subplot(3, 3, idx + 1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy())


def list_show_annotation_torch(dataloader):
    """
    Displays images and its annotations side by side from PyTorch DataLoader

    Args:
        dataloader (DataLoader): PyTorch dataloader
    """

    images, masks = next(iter(dataloader))

    images = images.numpy().transpose(0, 2, 3, 1)  # from (B,C,H,W) to (B,H,W,C)
    masks = masks.numpy().transpose(0, 2, 3, 1)  # from (B,num_classes,H,W) to (B,H,W,num_classes)

    plt.figure(figsize=(25, 15))
    plt.title("Images And Annotations")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

    # we set the number of image-annotation pairs to 9
    # feel free to make this a function parameter if you want
    n_images = min(9, images.shape[0])
    for idx in range(n_images):
        plt.subplot(3, 3, idx + 1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(images[idx], masks[idx])
