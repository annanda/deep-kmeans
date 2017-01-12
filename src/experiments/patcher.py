import cv2

from src.image_utils import draw_image, draw_images_with_matplot, get_random_patch_of_image, \
    get_random_patches_of_image


def main():
    img = cv2.imread("../../flower.jpg")
    draw_image(img)
    patch = get_random_patch_of_image(img, patch_width=165, patch_height=165)
    draw_image(patch)
    patch = get_random_patch_of_image(img, patch_width=16, patch_height=16)
    draw_image(patch)
    patches = get_random_patches_of_image(img, patch_width=100, patch_height=80, num_patches=25)
    draw_images_with_matplot(patches.reshape(5, 5, 80, 100, 3))


if __name__ == "__main__":
    main()
