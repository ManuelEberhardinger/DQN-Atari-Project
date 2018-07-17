import torchvision.transforms as T


class ImageProcessor:

    def __init__(self, size, device):
        self.transformations = T.Compose([T.ToPILImage(),
                                          T.Grayscale(),
                                          T.Resize(size),
                                          T.ToTensor()])
        self.device = device

    def process(self, screen):
        # crop the image to a square image
        screen = screen[34:-16, :, :]
        return self.transformations(screen).squeeze(0).to(self.device)
