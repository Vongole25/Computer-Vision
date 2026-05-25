from PIL import Image
from rembg import remove


class ImgWrapper:
    def __init__(self, height=224, width=224, channels=4):
        """
        Initializes the ImgWrapper class with specified image properties.

        :param height: The height to which images will be resized (default is 224).
        :param width: The width to which images will be resized (default is 224).
        :param channels: The number of channels for the image (default is 4).
        """
        self.height = height
        self.width = width
        self.channels = channels

    def __call__(self, img):
        """
        Processes the input image by resizing and cropping.

        :param img: The input PIL Image object.
        :return: The processed PIL Image object.
        """

        img = remove(img)
        padded_img = self.add_square_padding(img)
        resized_img = self.resize_image(padded_img)

        return resized_img

    def add_square_padding(self, img, color=(0, 0, 0)):
        """
        Adds a square padding to the input image.

        :param img: The input PIL Image object.
        :param color: The color of the square (default is (0, 0, 0)).
        :return: The processed PIL Image object.
        """
        width, height = img.size
        if width == height:
            return img  # if the img is square, return origin

        max_dim = max(width, height)
        new_img = Image.new('RGB', (max_dim, max_dim), color)

        new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        return new_img

    def remove_background(self, img):
        """
        Removes the background from the input image.
        :param img:
        :return:
        """

        return remove(img)

    def resize_image(self, img):
        """
        Resizes the image to the specified dimensions (width x height).

        :param img: The input PIL Image object to be resized.
        :return: The resized PIL Image object.
        """
        return img.resize((self.width, self.height), resample=Image.LANCZOS)

    def load_image(self, path):
        """
        Loads an image from the specified file path.

        :param path: The file path of the image to be loaded.
        :return: The loaded PIL Image object.
        """
        return Image.open(path)

    def show_image(self, img):
        """
        Displays the image using the default image viewer.

        :param img: The input PIL Image object to be displayed.
        :return: None.
        """
        img.show()

    def flip_image(self, img, mode='horizontal'):
        """
        Flips the image either horizontally, vertically, or both.

        :param img: The input PIL Image object to be flipped.
        :param mode: The mode of flipping, either 'horizontal', 'vertical', or 'both'.
        :return: The flipped PIL Image object.
        """
        if mode == 'horizontal':
            return img.transpose(Image.FLIP_LEFT_RIGHT)  # 수평 뒤집기
        elif mode == 'vertical':
            return img.transpose(Image.FLIP_TOP_BOTTOM)  # 수직 뒤집기
        elif mode == 'both':
            # 수평 및 수직으로 뒤집기
            return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("Invalid mode. Choose 'horizontal', 'vertical', or 'both'.")
