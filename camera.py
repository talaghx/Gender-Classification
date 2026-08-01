import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# اختيار الجهاز
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# إنشاء ResNet18 بنفس شكل المودل المستخدم في التدريب
model = models.resnet18(weights=None)

# تعديل آخر طبقة لتصنيف فئتين
model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# تحميل أفضل مودل تم تدريبه
checkpoint = torch.load(
    "best_gender_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

class_names = checkpoint["class_names"]

print("Classes:", class_names)

model = model.to(device)
model.eval()


# تجهيز صورة الوجه بنفس طريقة التدريب
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# تحميل كاشف الوجه
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Error: Face detector could not be loaded.")
    raise SystemExit


# فتح كاميرا اللابتوب
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera could not be opened.")
    raise SystemExit


# أقل نسبة ثقة لقبول التصنيف
CONFIDENCE_THRESHOLD = 75.0


while True:
    success, frame = camera.read()

    if not success:
        print("Error: Could not read camera frame.")
        break

    # تحويل الصورة إلى رمادي لكشف الوجه
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # كشف الوجوه
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for x, y, w, h in faces:

        # إضافة مساحة بسيطة حول الوجه
        padding = 20

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)

        x2 = min(
            x + w + padding,
            frame.shape[1]
        )

        y2 = min(
            y + h + padding,
            frame.shape[0]
        )

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # تحويل BGR إلى RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face_image = Image.fromarray(
            face_rgb
        )

        # تجهيز الصورة للمودل
        input_tensor = image_transform(
            face_image
        )

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(device)
        )

        # التنبؤ
        with torch.no_grad():
            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        confidence_percent = (
            confidence.item() * 100
        )

        predicted_class = class_names[
            predicted_index.item()
        ]

        # تحديد النص واللون
        if confidence_percent < CONFIDENCE_THRESHOLD:
            text = (
                f"Uncertain: "
                f"{confidence_percent:.1f}%"
            )

            box_color = (0, 165, 255)  # برتقالي

        else:
            text = (
                f"{predicted_class}: "
                f"{confidence_percent:.1f}%"
            )

            if predicted_class.lower() == "male":
                box_color = (255, 0, 0)  # أزرق

            elif predicted_class.lower() == "female":
                box_color = (255, 105, 180)  # وردي

            else:
                box_color = (0, 255, 0)  # أخضر احتياطي

        # رسم مربع حول الوجه
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        # كتابة التصنيف ونسبة الثقة
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2
        )

    cv2.imshow(
        "Gender Classification",
        frame
    )

    # اضغطي Q لإغلاق البرنامج
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()