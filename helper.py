results = []

def initialize_network():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_from_url(url):
    filename = "image.jpg"
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    return Image.open('image.jpg')

def process(image):
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0)

def printnorm(self, input, output):
    results.append(output.data)
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('***************************')

def register_hooks(model, layers):
    model_children = list(model.features.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            if counter in layers:
                model_children[i].register_forward_hook(printnorm)
    print(f"Total convolutional layers: {counter}")
