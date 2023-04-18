import cv2, sys 
import image_preprocessing as ip

def main(type):
    vidcap = cv2.VideoCapture(0)
    _, image = vidcap.read()
    for i in range(2000):

        resized_img = ip.pre_process(image)
        
        cv2.imwrite(f"./training_data/classification_training_data/{type}_frame{i}.jpg", resized_img) # save the img 
        cv2.imshow('frame',resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _, image = vidcap.read()
        
        print('Read a new frame: ', i)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python capture.py (blacklist or whitelist)")
        sys.exit(1)

    main(sys.argv[1])