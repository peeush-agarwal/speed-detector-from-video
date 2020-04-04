import cv2
import imutils
import os

def store_images_from_video(video_file_name, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_file_name)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        resized = imutils.resize(frame, height=300)
        cv2.imwrite(os.path.join(output_folder,'image'+str(i)+'.jpg'),resized)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_folder = './Data/train'
    output_folder = './Data/output'
    video_filename = 'train.mp4'
    text_filename = 'train.txt'
    store_images_from_video(os.path.join(input_folder, video_filename), output_folder)
    print('Images fetched from video:'+video_filename)