import cv2


class ColorSpace(object):
    def __init__(self):
        pass

    def value_range(self, image):
        raise NotImplementedError

    def pre_process_image(self, image):
        raise NotImplementedError

    def post_process_image(self, image):
        raise NotImplementedError

    def color_indices(self):
        raise NotImplementedError

    def light_indices(self,):
        raise NotImplementedError

    def is_not_black_region(self, image):
        channels = cv2.split(image)
        light_channels_sums = [cv2.sumElems(channels[j])[0] for j in self.light_indices()]
        return sum(light_channels_sums) > 50


class BGRColorSpace(ColorSpace):
    def __init__(self):
        pass

    def value_range(self, image):
        return 255

    def pre_process_image(self, image):
        return image.copy()

    def post_process_image(self, image):
        return image.copy()

    def color_indices(self):
        return [0, 1, 2]

    def light_indices(self,):
        return [0, 1, 2]

    def channels_count(self):
        return 3


class LABColorSpace(ColorSpace):
    def __init__(self):
        pass

    def value_range(self, image):
        return 255

    def pre_process_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def post_process_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    def color_indices(self):
        return [1, 2]

    def light_indices(self,):
        return [0]

    def channels_count(self):
        return 3


class HSVColorSpace(ColorSpace):
    def __init__(self):
        pass

    def value_range(self, image):
        return 255

    def pre_process_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def post_process_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    def color_indices(self):
        return [0,1]

    def light_indices(self,):
        return [2]

    def channels_count(self):
        return 3


class GrayscaleColorSpace(ColorSpace):
    def __init__(self):
        pass

    def value_range(self, image):
        return 255

    def pre_process_image(self, image):
        return image.copy()

    def post_process_image(self, image):
        return image.copy()

    def color_indices(self):
        return [0]

    def light_indices(self):
        return [0]

    def channels_count(self):
        return 1