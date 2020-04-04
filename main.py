from store_images_from_video import store_images_from_video
import os

if __name__ == "__main__":
    input_folder = './Data/train'
    output_folder = './Data/output'
    video_filename = 'train.mp4'
    text_filename = 'train.txt'
    store_images_from_video(os.path.join(input_folder, video_filename), output_folder)
    print('Images fetched from video:'+video_filename)