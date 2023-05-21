import cv2

if __name__ == '__main__':
    # Initialize Opencv DNN
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1 / 255)

    # Load Classes
    classes = []
    with open('classes.txt', "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)

    # Initialize Camera
    cap = cv2.VideoCapture('street.mp4')
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1280, 720))

    target_width = 800  # Specify the desired width of the displayed frame

    while True:
        # Get Frames
        ret, frame = cap.read()

        # Resize Frame to fit within the specified width
        frame = cv2.resize(frame, (target_width, int(frame.shape[0] * target_width / frame.shape[1])))

        # Object Detection
        class_ids, scores, boxes = model.detect(frame)
        if len(class_ids) > 0:
            for i in range(len(class_ids)):
                class_id = int(class_ids[i])
                score = scores[i]
                box = boxes[i]
                x, y, w, h = box
                cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 100 * class_id), 3)

        # Display Frames
        cv2.imshow("Frame", frame)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
