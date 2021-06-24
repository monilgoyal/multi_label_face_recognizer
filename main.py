from os import getenv, listdir
import cv2
import numpy as np
from definations import face_operation, whatsapp, mail, tf_aws
from sys import argv

# Loading credentials
from dotenv import load_dotenv
load_dotenv('credentials.env', override=True)

wa = whatsapp(account_sid=getenv('ACCOUNT_SID'),
              auth_token=getenv('AUTH_TOKEN'),
              tw_no=getenv('TWILIO_CONTACT')
              )

email = mail(
    email_from=getenv('EMAIL_FROM'),
    sendgrid_api_key=getenv('SENDGRID_API_KEY')
)
models = [m for m in listdir("./model/") if m[0] != '.']
for ind, ml in enumerate(models):
    print(f"{ind} : {ml}")
model_no = input("select your model: ") or argv[1]
try:
    if int(model_no) not in range(len(models)):
        print("resourse not listed")
    else:
        model = cv2.face_LBPHFaceRecognizer.create()
        model.read(f'./model/{models[int(model_no)]}')
        # Open Webcam
        cap = cv2.VideoCapture(0)

        while True:

            ret, image = cap.read()
            face = face_operation(image)
            try:
                # Pass face to prediction model
                label, conf = model.predict(face)
                label_info = model.getLabelInfo(label)
                if conf < 400:
                    confidence = int(100 * (1 - conf/400))
                    display_string = str(confidence) + \
                        f'% Confident u r {label_info}'

                cv2.putText(image, display_string, (100, 120),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)

                if confidence >= 80:
                    if label_info == "monil":
                        #                 wa.send("⚠️ *Monil Face Is Detected* ⚠️",getenv('RECV_NO'))
                        #                 email.send(email_to=getenv('EMAIL_TO'),subject="Face detected",content="User: Monil")
                        print(' monil')
                    elif label_info == "narayan":
                        #                 print(tf_aws("aws-key","ap-south-1","ami-010aff33ed5991201","t2.micro","5"))
                        print(' narayan')
                    break
                else:
                    cv2.putText(image, "looking for more..", (250, 450),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Face Recognition', image)

            except:
                cv2.putText(image, "No Face Found", (220, 120),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "looking for face", (250, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Recognition', image)
                pass

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break

        cap.release()
        cv2.destroyAllWindows()
except:
    print("Only integer is valid input")
