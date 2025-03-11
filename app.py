import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the trained model to classify signs
model = load_model('traffic_classifier.h5')

# Dictionary to label all traffic signs classes
classes = { 1: 'Giới hạn tốc độ (20 km/h)',
2: 'Giới hạn tốc độ (30 km/h)',
3: 'Giới hạn tốc độ (50 km/h)',
4: 'Giới hạn tốc độ (60 km/h)',
5: 'Giới hạn tốc độ (70 km/h)',
6: 'Giới hạn tốc độ (80 km/h)',
7: 'Kết thúc giới hạn tốc độ (80 km/h)',
8: 'Giới hạn tốc độ (100 km/h)',
9: 'Giới hạn tốc độ (120 km/h)',
10: 'Cấm vượt',
11: 'Cấm vượt xe có trọng tải trên 3.5 tấn',
12: 'Ưu tiên tại ngã tư',
13: 'Đường ưu tiên',
14: 'Nhường đường',
15: 'Dừng lại',
16: 'Cấm các loại phương tiện',
17: 'Cấm xe có trọng tải trên 3.5 tấn',
18: 'Cấm vào',
19: 'Chú ý chung',
20: 'Đường cong nguy hiểm bên trái',
21: 'Đường cong nguy hiểm bên phải',
22: 'Đường cong kép',
23: 'Đường gồ ghề',
24: 'Đường trơn',
25: 'Đường hẹp bên phải',
26: 'Công trình đường bộ',
27: 'Tín hiệu giao thông',
28: 'Người đi bộ',
29: 'Trẻ em băng qua đường',
30: 'Xe đạp băng qua',
31: 'Cẩn thận băng tuyết',
32: 'Động vật hoang dã băng qua',
33: 'Kết thúc giới hạn tốc độ và cấm vượt',
34: 'Rẽ phải phía trước',
35: 'Rẽ trái phía trước',
36: 'Chỉ đi thẳng',
37: 'Đi thẳng hoặc rẽ phải',
38: 'Đi thẳng hoặc rẽ trái',
39: 'Đi về phía bên phải',
40: 'Đi về phía bên trái',
41: 'Bắt buộc đi theo vòng xuyến',
42: 'Kết thúc lệnh cấm vượt',
43: 'Kết thúc lệnh cấm vượt xe có trọng tải trên 3.5 tấn'
,44:'CAM QUAY DAU XE' 
            }

# Initialise GUI
top = tk.Tk()
top.geometry('1000x1000')
top.title('Nhận Dạng Biển Báo')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = image.convert('RGB') 
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = np.argmax(model.predict([image]), axis=1)[0]
    sign = classes[pred + 1]
    print(f"{pred}: {sign}")
    label.configure(foreground='#011638', text=(f"{pred}: {sign}"))

def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận Dạng", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

from tkinter import filedialog
from PIL import Image, ImageTk

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        
        # Resize the image to a specific size (e.g., 128x128)
        specific_size = (300, 300)
        uploaded = uploaded.resize(specific_size)
        
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error: {e}")

img_path = "all.jpg"  # Đường dẫn tới ảnh của bạn
img = Image.open(img_path)  # Mở ảnh
img = img.resize((400, 200))  # Resize ảnh cho phù hợp với giao diện
photo = ImageTk.PhotoImage(img)  # Chuyển ảnh thành đối tượng có thể hiển thị trong tkinter

# Hiển thị ảnh trước tiêu đề
image_label = tk.Label(top, image=photo)
image_label.photo = photo  # Lưu tham chiếu đến ảnh (ngăn không bị thu hồi)
image_label.pack(pady=10) 

#
upload = Button(top, text="Tải Ảnh Lên", command=upload_image, padx=0, pady=0)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Kiểm Tra Biển Báo", pady=0, font=('arial', 20, 'bold'))

heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
