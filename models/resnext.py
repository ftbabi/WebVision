import pretrainedmodels
import torch
import pretrainedmodels.utils as utils

def test():
    model_name = 'resnext101_32x4d'  # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.eval()
    print(pretrainedmodels.model_names)
    load_img = utils.LoadImage()

    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(model)

    path_img = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/data/2018/img/google_images_resized/q10371/zzI9obeFLXNpZM.jpg'

    input_img = load_img(path_img)
    input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor,
                                    requires_grad=False)

    output_logits = model(input)  # 1x1000
    print(output_logits)

if __name__ == '__main__':
    test()