import cv2
from os import mkdir
from definations import face_operation
label = input("Give a label name: ")
mkdir(f'./faces/{label}')

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_operation(frame) is not None:
        count += 1
        face = cv2.resize(face_operation(frame), (200, 200))

        # Save file in specified directory with unique name
        file_name_path = f'./faces/{label}/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
