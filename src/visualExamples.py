from main import *
from PIL import ImageFont, ImageDraw, Image
import imageio
import numpy as np


def drawPredictedObjs(img_draw, pred, text_font, score_th=0.5):
    for i, objs in enumerate(pred["boxes"]):
        if pred["scores"][i] > score_th:
            img_draw.rectangle(objs.tolist(), outline="red", width=3)
            obj_text = "{}  {:1.3f}".format(exDark.get_label_name(pred["labels"][i]), pred["scores"][i])
            img_draw.text((objs.tolist()[0:2]), obj_text, fill="orange", font=text_font)


def drawGroundTruthObjs(img_draw, target):
    for i, gr_truth in enumerate(target["boxes"]):
        img_draw.rectangle(gr_truth.tolist(), outline="aqua", width=3)
        obj_text = "{}".format(exDark.get_label_name(target["labels"][i]))
        img_draw.text((gr_truth.tolist()[0:2]), obj_text, fill="white", font=font)


def im_resize(gifsize, img):
    im_w, im_h = img.size
    if (im_w / im_h) > 1:
        return img.resize((gifsize, int(gifsize * (im_h / im_w))))
    else:
        return img.resize((int(gifsize * (im_w / im_h)), gifsize))


if __name__ == "__main__":

    # Get Test Dataset
    exDark = ExDarkDataset()
    _, testdataset = getDataloaders(exDark, get_datasets=True)
    # Get Model
    model = get_RCNN_Model()
    model.load_state_dict(torch.load(r"..\models\model_1_epoch_14.pth"))
    # Set Font
    fontsize = 12
    font = ImageFont.truetype("arial.ttf", fontsize)
    num_imgs_in_gif = 50
    np.random.seed(0)
    rand_i = np.random.choice(len(testdataset), num_imgs_in_gif, replace=False)
    model.eval()
    # Initial image is for setting the gif size
    gif_size = 900
    gif_images = []
    gif_images.append(Image.new('RGB', (gif_size, gif_size)))
    for i in rand_i:
        images, targets = testdataset[i]
        predicitons = model(images.unsqueeze(0))
        im = torchvision.transforms.ToPILImage()(images)
        im1 = ImageDraw.Draw(im)
        drawPredictedObjs(im1, predicitons[0], font)
        drawGroundTruthObjs(im1, targets)
        im = im_resize(gif_size, im)
        gif_images.append(im)
        # im.show()
    imageio.mimsave(r"..\raw\sample.gif", gif_images, duration=1)
