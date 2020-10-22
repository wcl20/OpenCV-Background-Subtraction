import argparse
import cv2
import imutils
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to video")
    args = parser.parse_args()

    video = cv2.VideoCapture(args.input)

    # Video writer
    os.makedirs("output", exist_ok=True)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('M', "J", "P", "G")
    filename, _ = os.path.splitext(os.path.basename(args.input))
    writer = cv2.VideoWriter(f"output/{filename}_output.avi", fourcc, fps, size)

    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=25, detectShadows=False)
    # bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    area_thresh = 300

    while video.isOpened():

        success, frame = video.read()
        if success:
            # Resize frame
            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            # Foreground mask
            fg_mask = bg_subtractor.apply(blur)
            # Find contours
            contours = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            for contour in contours:
                # Filter contours
                if cv2.contourArea(contour) < area_thresh:
                    continue
                # Draw bounding boxes
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 3)

            # Display frame and save frame
            cv2.imshow("Frame", frame)
            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
