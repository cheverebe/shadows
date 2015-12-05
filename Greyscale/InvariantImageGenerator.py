import cv2
import numpy as np
import math
from LAB.shadow_detection.utils import show_and_save, equalize_hist_3d


class InvariantImageGenerator(object):

    def __init__(self):
        pass

    def get_invariant_image(self, image, angles=range(0, 179, 4)):
        two_dim_log_chrom = self.two_dim_log_chrom(image)
        min_entropy = -1
        out_image = None
        min_angle = -1
        for angle in angles:
            print(angle)
            one_d_projected_grayscale = self.project_into_one_d(two_dim_log_chrom, angle)

            #show_and_save("o1", 'madera', 'png', equalize_hist_3d(one_d_projected_grayscale))
            #show_and_save("o2", 'madera', 'png', o2)
            #show_image("img", one_d_projected_grayscale)
            entropy = self.calculate_entropy(one_d_projected_grayscale)
            if entropy < min_entropy or min_entropy == -1:
                min_entropy = entropy
                out_image = one_d_projected_grayscale
                min_angle = angle
        print("min angle: " + str(min_angle))
        return out_image

    def two_dim_log_chrom(self, image):
        log_chrom_image = self.log_chrom_image(image)
        print("Projection into 2D")
        two_d_log_chrom = self.project_to_2d(log_chrom_image)

        return np.uint8(two_d_log_chrom)

    def log_chrom_image(self, image):
        image = self.limit_image(image)
        (b_channel, g_channel, r_channel) = cv2.split(image)

        b_channel = np.float32(b_channel)
        g_channel = np.float32(g_channel)
        r_channel = np.float32(r_channel)

        prod = cv2.multiply(b_channel, g_channel)
        prod = cv2.multiply(prod, r_channel)
        cbrt = cv2.pow(prod, 1.0/3)

        b_channel = cv2.divide(b_channel, cbrt)
        g_channel = cv2.divide(g_channel, cbrt)
        r_channel = cv2.divide(r_channel, cbrt)

        b_channel = cv2.log(b_channel)
        g_channel = cv2.log(g_channel)
        r_channel = cv2.log(r_channel)
        log_chrom_image = cv2.merge((b_channel, g_channel, r_channel))
        show_and_save("chrom_image", 'log', 'png', equalize_hist_3d(log_chrom_image))
        return log_chrom_image

    def limit_image(self, image):
        maxval = 245
        minval = 10

        retval, inverted_mask = cv2.threshold(image, maxval, 255, cv2.THRESH_BINARY_INV)
        # erase mask values
        mask = 255 - inverted_mask
        image = cv2.bitwise_and(image, inverted_mask)
        image += (mask / 255) * maxval

        retval, inverted_mask = cv2.threshold(image, minval, 255, cv2.THRESH_BINARY)
        # erase mask values
        mask = 255 - inverted_mask
        image = cv2.bitwise_and(image, inverted_mask)
        image += (mask / 255) * minval

        return image

    def get_U(self):
        return np.matrix([[1.0/math.sqrt(2), -1.0/math.sqrt(2), 0], [1.0/math.sqrt(6), 1.0/math.sqrt(6), -2.0/math.sqrt(6)]])

    def get_U_1(self):
        return np.array([1.0/math.sqrt(2), -1.0/math.sqrt(2), 0])

    def get_U_2(self):
        return np.array([1.0/math.sqrt(6), 1.0/math.sqrt(6), -2.0/math.sqrt(6)])

    def project_to_2d(self, log_chrom):
        U = self.get_U()
        #B = [[np.array((U * np.matrix(elem).transpose())) for elem in row] for row in log_chrom]
        U1 = self.get_U_1()
        U2 = self.get_U_2()
        out_1 = U1 * log_chrom
        out_2 = U2 * log_chrom
        (c1, c2, c3) = cv2.split(out_1)
        out_1 = c1 + c2 + c3
        (c1, c2, c3) = cv2.split(out_2)
        out_2 = c1 + c2 + c3
        #show_and_save("chrom_image", '2d_log_1', 'png', equalize_hist_3d(np.array(out_1)))
        #show_and_save("chrom_image", '2d_log_2', 'png', equalize_hist_3d(np.array(out_2)))
        A = cv2.merge([out_1, out_2])
        return A

    @staticmethod
    def project_into_one_d(two_dim_log_chrom, angle):
        rad = math.radians(angle)

        (d1, d2) = cv2.split(two_dim_log_chrom)
        r1 = d1 * math.cos(rad)
        r2 = d2 * math.sin(rad)
        return r1 + r2

    def calculate_entropy(self, mono):
        #return entropy(mono)
        frame = equalize_hist_3d(mono)
        #hist = cv2.calcHist([mono], [0], None, [bins], [0, 256])
        histogram = np.histogram(frame, bins=64)[0]
        histogram_length = sum(histogram)
        samples_probability = [float(h) / histogram_length for h in histogram]
        entropy = -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

        return entropy