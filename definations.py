import cv2
from twilio.rest import Client
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from subprocess import getstatusoutput
from twilio.rest import Client
from sendgrid.helpers.mail import Mail
from sendgrid import SendGridAPIClient

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_operation(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        crop_img = gray[y:y+h, x:x+w]
    return crop_img


class whatsapp:
    """
    :param str account_sid : twilio account_sid
    :param str auth_token  : twilio auth_token
    """
    def __init__(self,account_sid=None,auth_token=None,tw_no=None): 
        self.client = Client(account_sid, auth_token)
        self.tw_no = tw_no
    def send(self, text=None, recv_no=None):
        """
        :param str text    : message to send 
        :param str recv_no : receiver number with country code
        """
        message = self.client.messages.create(from_=f'whatsapp:{self.tw_no}',body=text,to=f'whatsapp:{recv_no}')
        return message.sid


class mail:
    """
    :param str sendgrid_api_key : provided by sendgrid
    """
    def __init__(self,email_from=None,sendgrid_api_key=None): 
        self.client = SendGridAPIClient(sendgrid_api_key)
        self.email_from= email_from
    def send(self,email_to=None,subject=None,content=None):
        try:
            response = self.client.send(message = Mail(
                                            from_email=self.email_from,
                                            to_emails= email_to,
                                            subject=subject,
                                            html_content=content
                                            )
                                       )
            if(response.status_code==202):
                return 'successfully send'
            else:
                return 'some thing wrong happened'
        except Exception as e:
            return e.message
        
def tf_aws(key_name,region,ami,instance_type,ebs_size):
    print("Creating Infrastructure....")
    status,output=getstatusoutput(f'terraform -chdir="./terraform/aws" init && \
    terraform -chdir="./terraform/aws" apply \
    -var key_name={key_name} \
    -var ami={ami} \
    -var instance_type={instance_type} \
    -var ebs_size={ebs_size} \
    -var region={region}\
    -auto-approve ')
    if status==0:
        return "Infrastructure created successfully.."
    else:
        return "Failed!!..Wrong details provided"