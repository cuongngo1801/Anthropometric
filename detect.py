from tkinter import *
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import mysql.connector
import os
import csv
from PIL import Image, ImageTk
import time
import cv2
import numpy as np
import pandas as pd 
import math 
import csv
from dimention import *
from dimention import LineBuilder
import glob
import threading

# def open_image():
# 	root.filename = filedialog.askopenfilename(initialdir = 'C:/Users/Ngo Cuong/Desktop/Project/YOLO', title = ' Open Image', filetypes = (('jpg files','*.jpg'),('png files','*.png'), ('all files','*.*'))) 
# 	label = Label(root, textvariable = root.filename).pack(side = tk.LEFT)

def open_camera():
	variable = 'C:/Program Files (x86)/Canon/EOS Utility/EOS Utility.exe'
	os.system('"%s"' %variable)
def thread_camera():
	threading.Thread(target =open_camera).start()


def thread_measurement():
	threading.Thread(target =insomething).start()

def insomething():
	# image = root.filename
	sexual = gender.get()
	ID = e1.get()
	# direction = listroom.get()
	# cursor.execute("CREATE TABLE medical(ID INTEGER AUTO_INCREMENT PRIMARY KEY, Comment CHAR(255) NOT NULL, Parameter FLOAT NOT NULL)")
	config = './config/yolov4-custom.cfg'
	weights = './config/yolov4-custom_last.weights'
	classes = './config/yolo.names'
	def get_output_layers(net):
	    layer_names = net.getLayerNames()

	    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	    return output_layers


	def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
	    label = str(classes[class_id])

	    color = COLORS[class_id]

	    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

	    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# Angle and diamention caculator.
	def distance_caculation2P (point_1, point_2):
	    x1, y1 = point_1
	    x2, y2 = point_2
	    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
	    return dist

	def angle_caculation(point_1, point_2, point_3):
	    P21 = point_1 - point_2
	    P23 = point_3 - point_2
	    cos_angle = np.dot(P21,P23)/(np.linalg.norm(P21)*np.linalg.norm(P23))
	    angle = np.arccos(cos_angle)
	    angle = np.degrees(angle)
	    return angle
	def distance (point_1, point_2):
	    dist = abs(point_2 - point_1)
	    return dist


	path = glob.glob("./Image/*.JPG")

	data = []

	for img in path:
		images = cv2.imread(img)
		D = 0
		rate = 1
		image1 = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
		fig = plt.figure(figsize=(12, 9))
		ax = fig.add_subplot()
		ax.xaxis.tick_top()
		plt.imshow(image1)
		cv2.destroyAllWindows()
		ax.set_title('click to build line segments')
		lines, = ax.plot([0], [0])  # empty line // declare lines is type Line 2D in plt
		a = LineBuilder(lines, rate,D)
		plt.show()

		data.append(a.value())
		print('gia tri cua D ne',data)


	count = 0

	with open(classes, 'r') as f:
	    classes = [line.strip() for line in f.readlines()]

	for lst in path:

		image = cv2.imread(lst)
		Width = image.shape[1]
		Height = image.shape[0]

		scale = 0.00392
		x = 0
		y = 1

		COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

		net = cv2.dnn.readNet(weights, config)

		blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

		net.setInput(blob)

		outs = net.forward(get_output_layers(net))
		# print (np.shape(outs))


		center_box= []
		class_ids = []
		confidences = []
		boxes = []
		conf_threshold = 0.5
		nms_threshold = 0.4

		# Thực hiện xác định bằng HOG và SVM
		start = time.time()
		i=0
		asw = 0
		for out in outs:
		    asw= asw+1
		    for detection in out:
		        i = i+1
		        scores = detection[5:]
		        class_id = np.argmax(scores)
		        confidence = scores[class_id]
		        if confidence > 0.5:
		            center_x = int(detection[0] * Width)
		            center_y = int(detection[1] * Height)
		            w = int(detection[2] * Width)
		            h = int(detection[3] * Height)
		            x = center_x - w / 2
		            y = center_y - h / 2
		            class_ids.append(class_id)

		            center_box.append ([center_x, center_y])

		            # print("class ne",class_ids)
		            confidences.append(float(confidence))
		            boxes.append([x, y, w, h])

		            # print(boxes)
		            # print(center_box)
		    # print ("i là :",i)
		# print ("asư là :",asw)
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
		# print("indices là :",indices)
		en,k,tr,prn = [],[],[],[]
		mf,al,ch,n = [],[],[],[]
		num14, num15, num4 = [],[],[0,0]
		num17,num1,num2=[],[],[]
		num13,num16, num10 = [0,0],[0,0],[0,0]

		for i in indices:
		    i = i[0]
		    box = boxes[i]
		    x = box[0]
		    y = box[1]
		    w = box[2]
		    h = box[3]
		    if class_ids[i] == 0:
		        num1 = np.array(center_box[i])
		    elif class_ids[i] == 1:
		        num2 = np.array(center_box[i])
		    elif class_ids[i] == 2:
		        num3 = np.array(center_box[i])
		    elif class_ids[i] == 3:
		        num4 = np.array(center_box[i])
		    elif class_ids[i] == 4:
		        num5 = np.array(center_box[i])
		    elif class_ids[i] == 5:
		        num6 = np.array(center_box[i])
		    elif class_ids[i] == 6:
		        num7 = np.array(center_box[i])
		    elif class_ids[i] == 7:
		        num8 = np.array(center_box[i])      
		    elif class_ids[i] == 8:
		        num9 = np.array(center_box[i])
		    elif class_ids[i] == 9:
		        num10 = np.array(center_box[i])
		    elif class_ids[i] == 10:
		        num11 = np.array(center_box[i])
		    elif class_ids[i] == 11:
		        num12 = np.array(center_box[i])
		    elif class_ids[i] == 12:
		        num13 = np.array(center_box[i])
		    elif class_ids[i] == 13:
		        num14 = np.array(center_box[i])
		    elif class_ids[i] == 14:
		        num15 = np.array(center_box[i])
		    elif class_ids[i] == 15:
		        num16 = np.array(center_box[i])
		    elif class_ids[i] == 16:
		        num17.append(center_box[i])
		    elif class_ids[i] == 17:
		        tr = np.array(center_box[i])
		    elif class_ids[i] == 18:
		        g = np.array(center_box[i])
		    elif class_ids[i] == 19:
		        en.append(center_box[i])

		    elif class_ids[i] == 20:
		        n = np.array(center_box[i])
		    elif class_ids[i] == 21:

		        mf.append(center_box[i])
		    elif class_ids[i] == 22:
		        al.append(center_box[i])

		    elif class_ids[i] == 23:
		        prn = np.array(center_box[i])
		    elif class_ids[i] == 24:
		        sn = np.array(center_box[i])
		    elif class_ids[i] == 26:
		        ls = np.array(center_box[i])
		    elif class_ids[i] == 27:
		        gn = np.array(center_box[i])
		    elif class_ids[i] == 28:
		        Ruler = np.array(center_box[i])
		    elif class_ids[i] == 25:
		        ch.append(center_box[i])

		    elif class_ids[i] == 31:
		        k = np.array(center_box[i])
		    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
		    # compare_value(i)

		if np.shape(num17) == (2,2):
		    num17_1 = np.array(num17[0])
		    num17_2 = np.array(num17[1])
		else: num17_1 = np.array(num17)

		if np.shape(ch) == (2,2):
		    ch1 = np.array(ch[0])
		    ch2 = np.array(ch[1])
		else: ch1 = np.array(ch)

		# # print("gia tri ch", ch1[0,1])

		# cv2.imshow("object detection", image)



		# cv2.waitKey()
		# cv2.destroyAllWindows()

		# Tạo các điểm ở khung detection, learning what is conference and how can YOLO work ?
		# Tạo các kẻ các đường thẳng, tính góc ở tâm của các detection.

		dn_prn,dmf, dal, dsn_prn, d_en, dn_sn, d1_2 = None, None, None, None, None, None, None
		g_n_prn, n_prn_gn, g_sn_gn, n_prn_sn, prn_sn_ls, n_13_sn, sn_13_gn, prn_n_sn, dsn_gn =None, None, None, None, None, None,None, None, None
		dch_gn, RP_GP, AB, AC, AB_1, D_AB, AB_2, D_AB2, dtr_n = None, None, None, None, None, None,None, None, None
		n_en,n_mf = '',''

		try:
			# Chiều dài sống mũi: Khoảng cách giữa điểm gốc mũi (n) và điểm đỉnh mũi trên da (prn)
			dn_prn = distance(n[1],prn[1])*data[count]

			# Độ nhô đỉnh mũi:Khoảng cách giữa điểm dưới mũi (sn) và điểm đỉnh mũi trên da (prn)
			dsn_prn = distance (sn[1],prn[1])*data[count]

			# Chiều cao mũi: Khoảng cách giữa điểm gốc mũi (n) và điểm dưới mũi (sn)
			dn_sn = distance (n[1],sn[1])*data[count]
		except:
			print('Khong co du du lieu')

		# ĐO TRỰC TIẾP
		if (np.shape(num17) == (2,2) or np.shape(mf) == (2,2) or np.shape(al) == (2,2)): 
		# Vị trí của điểm gốc mũi (n) so với điểm hàm trán (mf)
		    if n[1] > mf[0][1] == True:
		        n_mf = 'Up'
		    else: n_mf = 'Down'
		# Vị trí của điểm gốc mũi (n) so với điểm khóe mắt trong (en)

		# Khoảng cách gốc mũi: Khoảng cách giữa hai điểm hàm trán (mf)
		    if np.shape(mf) == (2,2):
		        mf1 = np.array(mf[0])
		        mf2 = np.array(mf[1])
		        dmf = distance(mf1[0],mf2[0])*data[count]
		    else: mf1 = np.array(mf)

		# Vị trí của điểm gốc mũi (n) so với điểm khóe mắt trong (en)
		# Khoảng cách gian khóe mắt trong (en)
		    if np.shape(en) == (2,2):
		        en1 = np.array(en[0])
		        en2 = np.array(en[1])
		        if en1[1] > n[1]:
		        	n_en = 'Up'
		        else: n_en= 'Down'
		        d_en = distance(en1[0],en2[0])*data[count]
		    elif  np.shape(en) == (2,):
		        en1 = np.array(en)
		        if n[1] < en1[1]:
		            n_en ='Up'
		        else: n_en = 'Down'

		    # Mũi có điểm gù xương (k) không
		    if k == []:
		        dk='Withdown K'
		        print('diem n',n,'diem en1',en)


		# Chiều rộng mũi mô mềm: Khoảng cách giữa hai điểm cánh mũi (al)
		    if np.shape(al) == (2,2):
		        al1 = np.array(al[0])
		        al2 = np.array(al[1])
		        dal = distance (al1[0],al2[0])*data[count]
		    else: al1 = np.array(al)
		    print ('Chiều cao mũi(n_sn)',dn_sn,'Chiều dài sống mũi(n_prn)',dn_prn,'Độ nhô đỉnh mũi(sn_prn)',
		    	dsn_prn,'Chiều rộng mũi mô mềm (al_al)',dal,'Chiều dài đoạn en',d_en,'chiều dài mf',dmf)
		    cv2.imwrite("./Image_detection/"+str(count)+'_'+ID+'_'+'Front'+'_'+sexual+".jpg", image)
		    a=-1

		# ĐO HÌNH CẠNH
		if np.shape(num14) == (2,) or np.shape(num15) == (2,) or np.shape(num1) == (2,) :
		# Tai phải
		    if num14[0] < ls[0] or num4[0] < num17_1[0][0] or ls[0] > num10[0]:

		    	try:
		    		# Chiều cao trán: Khoảng cách từ điểm giữa đường chân tóc (tr) đến điểm gốc mũi (n)
		    		dtr_n = distance (tr[1],n[1])*data[count]
		    		print('dtr_n',dtr_n)
		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    		dn_prn = distance_caculation2P(n,prn)*data[count]
		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	# Góc mũi trán (1): g – n – prn
		        	g_n_prn = angle_caculation (g,n,prn)

		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	# Góc mũi trán (1): g – n – prn
		        	g_n_prn = angle_caculation (g,n,prn)
		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	# Góc mũi trán (1): g – n – prn
		        	g_n_prn = angle_caculation (g,n,prn)
		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	# Góc mũi cằm (3): n – prn – gn
		        	n_prn_gn = angle_caculation (n,prn,gn)
		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	# Góc (5): n – prn – sn
		        	n_prn_sn = angle_caculation (n,prn,sn)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc mũi môi (6): prn – sn – ls
		        	prn_sn_ls = angle_caculation (prn,sn,ls)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc (9): sn – 8 – gn
		        	sn_13_gn = angle_caculation (sn,num13,gn)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc mũi mặt: prn – n – sn
		        	prn_n_sn = angle_caculation (prn,n,sn)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc (4): g – sn – gn
		        	g_sn_gn = angle_caculation (g,sn,gn)

		    	except Exception as e:
		    		print('Lỗi :',e)	
		    	try:
		    	# Góc (8): n – 8 – sn
		        	n_13_sn = angle_caculation (n,num13,sn)

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	# Chiều cao tầng mặt dưới: Khoảng cách từ điểm dưới mũi (sn) đến điểm cằm (gn)
		        	dsn_gn = distance (sn[1],gn[1])*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	 # Đo khoảng cách ch - gn
		        	dch_gn = distance(ch1[0,1],gn[1])*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	 # Đo khoảng cách ch - gn
		        	dch_gn = distance(ch1[0,1],gn[1])*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)	
		    	try:
		    	  # print(dn_prn)
		        	d1_2 = distance(num1,num2)*data[count]
		        	print('tai 1 2',d1_2)
		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	  # Khoảng cách giữa đường tiếp tuyến đi qua mí trên (CP) và đường tiếp tuyến đi qua đồng tử (CoP)
		        # if np.shape(en) != (2,2):
		        #     CP_Cop = distance(en1[x],)
		        # Khoảng cách giữa đường tiếp tuyến đi qua điểm gốc mũi (RP) và đường tiếp tuyến đi qua điểm glabella (GP)

			        # Từ điểm gốc mũi n (C) kẻ vuông góc với đường thẳng ngang (AB) kẻ từ điểm đỉnh mũi prn (B), điểm giao nhau (A) tại rãnh cánh mũi. Tính khoảng cách AB và AC
			        AB=abs(np.cross(al-n,prn-n)/np.linalg.norm(al-n))*data[count]
			        AC=abs(np.cross(prn-al,n-al)/np.linalg.norm(prn-al))*data[count]
			        RP_GP = distance (n[1],g[1])*data[count]

			        # Khoảng cách từ điểm gốc mũi n (A) đến điểm da môi trên ls (B)
			        AB_1 = distance_caculation2P(n,ls)*data[count]

			        # Khoảng cách từ điểm đỉnh mũi prn (D) đến hình chiếu vuông góc của nó trên đường AB
			        D_AB = abs(np.cross(ls-n,prn-n)/np.linalg.norm(ls-n))*data[count]

			        # Khoảng cách từ điểm gốc mũi n (A) chạy tiếp tuyến với nếp rãnh cánh mũi và tận cùng chấm dứt tại bờ xương hàm dưới (B)
			        AB_2 = distance_caculation2P(n,gn)*data[count]

			        # Khoảng cách từ điểm đỉnh mũi prn (D) đến hình chiếu vuông góc của nó trên đường AB
			        D_AB2 = abs(np.cross(gn-n,prn-n)/np.linalg.norm(gn-n))*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	cv2.imwrite("./Image_detection/"+str(count)+'_'+ID+'_'+'Right'+'_'+sexual+".jpg", image)
		    	a=0

		#Tai trái
		    if num14[0] > ls[0] or num4[0] > num17_1[0][0] or num10[0] > ls[0]:
		    	try:
		    		# Chiều cao trán: Khoảng cách từ điểm giữa đường chân tóc (tr) đến điểm gốc mũi (n)
		    		dtr_n = distance (tr[1],n[1])*data[count]
		    		print('dtr_n',dtr_n)
		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    		dn_prn = distance_caculation2P(n,prn)*data[count]
		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	# Góc mũi trán (1): g – n – prn
		        	g_n_prn = angle_caculation (g,n,prn)

		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	# Góc mũi trán (1): g – n – prn
		        	g_n_prn = angle_caculation (g,n,prn)
		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	# Góc mũi trán (1): g – n – prn
		        	g_n_prn = angle_caculation (g,n,prn)
		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	# Góc mũi cằm (3): n – prn – gn
		        	n_prn_gn = angle_caculation (n,prn,gn)
		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	# Góc (5): n – prn – sn
		        	n_prn_sn = angle_caculation (n,prn,sn)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc mũi môi (6): prn – sn – ls
		        	prn_sn_ls = angle_caculation (prn,sn,ls)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc (9): sn – 8 – gn
		        	sn_13_gn = angle_caculation (sn,num13,gn)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc mũi mặt: prn – n – sn
		        	prn_n_sn = angle_caculation (prn,n,sn)
		    	except Exception as e:
		    		print('Lỗi :',e)		    	
		    	try:
		    	# Góc (4): g – sn – gn
		        	g_sn_gn = angle_caculation (g,sn,gn)

		    	except Exception as e:
		    		print('Lỗi :',e)	
		    	try:
		    	# Góc (8): n – 8 – sn
		        	n_13_sn = angle_caculation (n,num13,sn)

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	# Chiều cao tầng mặt dưới: Khoảng cách từ điểm dưới mũi (sn) đến điểm cằm (gn)
		        	dsn_gn = distance (sn[1],gn[1])*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	 # Đo khoảng cách ch - gn
		        	dch_gn = distance(ch1[0,1],gn[1])*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	try:
		    	 # Đo khoảng cách ch - gn
		        	dch_gn = distance(ch1[0,1],gn[1])*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)	
		    	try:
		    	  # print(dn_prn)
		        	d1_2 = distance(num1,num2)*data[count]
		        	print('tai 1 2',d1_2)
		    	except Exception as e:
		    		print('Lỗi :',e)

		    	try:
		    	  # Khoảng cách giữa đường tiếp tuyến đi qua mí trên (CP) và đường tiếp tuyến đi qua đồng tử (CoP)
		        # if np.shape(en) != (2,2):
		        #     CP_Cop = distance(en1[x],)
		        # Khoảng cách giữa đường tiếp tuyến đi qua điểm gốc mũi (RP) và đường tiếp tuyến đi qua điểm glabella (GP)

			        # Từ điểm gốc mũi n (C) kẻ vuông góc với đường thẳng ngang (AB) kẻ từ điểm đỉnh mũi prn (B), điểm giao nhau (A) tại rãnh cánh mũi. Tính khoảng cách AB và AC
			        AB=abs(np.cross(al-n,prn-n)/np.linalg.norm(al-n))*data[count]
			        AC=abs(np.cross(prn-al,n-al)/np.linalg.norm(prn-al))*data[count]
			        RP_GP = distance (n[1],g[1])*data[count]

			        # Khoảng cách từ điểm gốc mũi n (A) đến điểm da môi trên ls (B)
			        AB_1 = distance_caculation2P(n,ls)*data[count]

			        # Khoảng cách từ điểm đỉnh mũi prn (D) đến hình chiếu vuông góc của nó trên đường AB
			        D_AB = abs(np.cross(ls-n,prn-n)/np.linalg.norm(ls-n))*data[count]

			        # Khoảng cách từ điểm gốc mũi n (A) chạy tiếp tuyến với nếp rãnh cánh mũi và tận cùng chấm dứt tại bờ xương hàm dưới (B)
			        AB_2 = distance_caculation2P(n,gn)*data[count]

			        # Khoảng cách từ điểm đỉnh mũi prn (D) đến hình chiếu vuông góc của nó trên đường AB
			        D_AB2 = abs(np.cross(gn-n,prn-n)/np.linalg.norm(gn-n))*data[count]

		    	except Exception as e:
		    		print('Lỗi :',e)
		    	cv2.imwrite("./Image_detection/"+str(count)+'_'+ID+'_'+'Left'+'_'+sexual+".jpg", image)
		    	a =1

		# In file csv:
		if (np.shape(num17) == (2,2) or np.shape(mf) == (2,2) or np.shape(al) == (2,2)): 
		    comment = ['diem n so voi diem mf','diem n so voi diem en', 'Co diem k hay khong','Chieu cao mui(n_sn)','Chieu dai song mui(n_prn)',
		    'Do nho dinh mui (sn_prn)','Chieu rong mo mem (al_al)','Chieu dai doan en','Chieu dai doan mf']
		    comment_count = 0
		    Parameter = (0,n_mf,n_en,'None',dn_sn,dn_prn,dsn_prn,dal,d_en,dmf)
		    with open("./File CSV/"+str(count)+'_'+ID+"Front"+sexual+'.csv','w',newline= '') as f:
		        fieldnames = ['Comment','Parameter']
		        thewiter = csv.DictWriter(f, fieldnames= fieldnames)
		        thewiter.writeheader()
		        for Comment in comment:	
		            comment_count +=1
		            thewiter.writerow({'Comment':Comment,'Parameter':Parameter[comment_count]}) 

		else:
		    comment = ["dn_prn", "dn_sn", "dsn_prn", "g_n_prn",'n_prn_gn','g_sn_gn','n_prn_sn','prn_sn_ls','n_13_sn','sn_13_gn',
		    'prn_n_sn','dsn_gn','dch_gn','RP_GP','AB','AC','AB_1','D_AB','AB_2','D_AB2','dtr_n','d1_2']
		    comment_count = 0
		    Parameter = (0,dn_prn, dn_sn, dsn_prn, g_n_prn,n_prn_gn,g_sn_gn,n_prn_sn,prn_sn_ls,n_13_sn,sn_13_gn,prn_n_sn,dsn_gn,
		        dch_gn,RP_GP,AB,AC,AB_1,D_AB,AB_2,D_AB2,dtr_n,d1_2)
		    if a == 1:
		    	file =str(count)+'_'+ID+'Left'+sexual+'.csv'
		    elif a ==0:
		    	file = str(count)+'_'+ID +'Right'+sexual+'.csv'
		    else: print("None")
		    with open("./File CSV/"+file,'w',newline= '') as f:
		        fieldnames = ['Comment','Parameter']
		        thewiter = csv.DictWriter(f, fieldnames= fieldnames)
		        thewiter.writeheader()
		        for Comment in comment:	
		            comment_count +=1
		            thewiter.writerow({'Comment':Comment,'Parameter':Parameter[comment_count]})

		end = time.time()
		print("YOLO Execution time: " + str(end-start))
		count = count + 1

root = Tk()
root.geometry('500x560')
root.maxsize(500,560)
root.minsize(500,450)
root.iconbitmap("./Ico/medical.ico")
root.title("Profile document")


mydb = mysql.connector.connect(
	host='127.0.0.1',
	user ='root',
	password ='19001560',
	database= 'testting',
	)

# print(mydb)
cursor =mydb.cursor()


def open_new_window():
	
	mydata = []

	def update(rows):
		global mydata
		mydata = rows 
		trv.delete(*trv.get_children())
		for i in rows:
			trv.insert('', 'end', values =i)

	def search():
		q2 =q.get()
		query = "SELECT ID, Comment, Parameter FROM medical WHERE Comment LIKE '%"+q2+"%' OR Parameter LIKE'%"+q2+"%' OR ID LIKE '%"+q2+"%'"
		cursor.execute(query)
		rows = cursor.fetchall()
		update(rows)

	def clear():
		query = "SELECT ID, Comment, Parameter FROM medical"
		cursor.execute(query)
		rows = cursor.fetchall()
		update(rows)

	def export():
		if len(mydata) < 1:
			messagebox.showerror("No Data", "No data available to export")
			return False
		fln = filedialog.asksaveasfilename(initialdir = os.getcwd(), title = "Save CSV", filetypes = (("CSV File", "*.csv"),("All Files","*.*")))
		with open(fln, mode = 'w', newline = '') as f:
			exp_writer = csv.writer(f, delimiter=',')
			for i in mydata:
				exp_writer.writerow(i)
		messagebox.showinfo("Data Exported", "You has been exported to "+ os.path.basename(fln)+" successfully.")

	def Import():
		mydata.clear()
		fln = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Open CSV", filetypes = (("CSV File", "*.csv"),("All Files","*.*")))
		with open (fln) as f:
			csvread = csv.reader(f, delimiter = ',')
			for i in csvread:
				mydata.append(i)
		update(mydata)

	def savedb():
		if messagebox.askyesno("Confirmation","Are you sure you want to save data to database"):
			for i in mydata:
				uid = i[0]
				comment = i[1]
				parameter = i[2]
				query = "INSERT INTO medical(ID, Comment, Parameter ) VALUES (NULL, %s, %s)"
				cursor.execute(query, (comment,parameter))
			mydb.commit()
			clear()
			messagebox.showinfo("Data Save", "Data has been save to database")
		else:
			return False
	

	top = Toplevel()
	top.geometry('700x600')
	top.title("My project")
	q = StringVar()

	# Make Tree View Table
	wrapper1 = LabelFrame(top, text= "Customer List")
	wrapper2 = LabelFrame(top, text = "Search")

	wrapper1.pack(fill="both", expand="yes", padx = 20, pady = 10)
	wrapper2.pack(fill="both", expand='yes', padx = 20, pady = 10)

	trv =ttk.Treeview(wrapper1, columns = (1,2,3), show = "headings", height = "18")
	trv.pack()

	trv.heading(1, text = "Customer ID")
	trv.heading(2, text = "Comment")
	trv.heading(3, text = "Parameter")


	expbtn = Button(wrapper1, text = "Export CSV", command = export)
	expbtn.pack(side= tk.LEFT, padx = 10, pady = 10) 

	impbtn = Button(wrapper1, text = "Import CSV", command = Import)
	impbtn.pack(side= tk.LEFT, padx = 10, pady = 10) 

	savebtn = Button(wrapper1, text = "Save Data", command = savedb)
	savebtn.pack(side= tk.LEFT, padx = 10, pady = 10)

	exitbtn = Button(wrapper1, text = 'Exit', command = top.destroy)
	exitbtn.pack(side= tk.LEFT, padx = 10, pady = 10)


	query = "SELECT ID, Comment, Parameter FROM medical"
	cursor.execute(query)
	rows = cursor.fetchall()
	update(rows)

	# Search Section
	lbl = Label(wrapper2, text = "Search")
	lbl.pack(side = tk.LEFT, padx = 10)
	ent = Entry(wrapper2, textvariable = q)
	ent.pack(side = tk.LEFT,padx = 6)
	btn = Button(wrapper2, text = "Search", command = search)
	btn.pack(side= tk.LEFT, pady = 6)
	cbtn =Button(wrapper2,text = "Clear", command = clear)
	cbtn.pack(side = tk.LEFT, pady= 6 )



img = Image.open("./Ico/profile.png")
img = img.resize((100,100))


my = ImageTk.PhotoImage(img)
label = Label(root, image = my)
label.place (x = 200, y = 10)

l1 = Label(root,text="Profile table ",font= "time 20 bold")
l1.place(x=165, y=120)

e1 = StringVar()
l2 = Label(root,text= "Enter ID", font = "time 15 bold")
l2.place(x=30, y= 220)
e1 = Entry (root, width = 30, bd = 3)
e1.place(x = 240,y = 220)

l4 = Label(root, text= "Select your Gender", font = "time 15 bold")
l4.place(x = 30, y=270)

gender = StringVar()
g1 = Radiobutton(root, text = "Male", variable = gender, value = "Male", font = 'time 15')
g1.select()
g1.place(x = 240, y =270)

g2 = Radiobutton(root, text = "Female", variable = gender, value = "Female", font = 'time 15')
g2.select()
g2.place(x = 240, y =300)


button = Button(root, text = "Take picture",command = thread_camera, fg = "white",bg = "green", font = "time 15 bold")
button.place(x= 40, y = 370)

button = Button(root, text = "Make file Excel", command = thread_measurement, fg = "white",bg = "blue", font = "time 15 bold")
button.place(x= 300, y = 370)

button = Button(root, text = "Save data", command = open_new_window, fg = "white",bg = "red", font = "time 15 bold", width = 34)
button.place(x= 40, y = 450)



root.mainloop()


# my_cursor.execute("CREATE TABLE users (name varchar(255), email varchar(255), age integer (10), users_id integer AUTO_INCREMENT PRIMARY KEY)")