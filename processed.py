# -*- coding: utf-8 -*-

import cv2
import numpy as np
import imutils

from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import os
import datetime
import json


SAVE_ERROR_DATA = True

found_circles = None
img_color = 0   # 0 = greyscale, 1 = color
img = None
blurred_img = None
img_path = None
final_image = None
min_radius = 10
max_radius = 50
min_distance_between_circles = 50
highlight_color = [255,0,0] # Green
white_color_pixel = 255
radios = [] # We will save the radios of the detected circles
hough_radius = []
drawn_radius = []
original_image_x = 0
original_image_y = 0
resize_x = 800
resize_y = 600
image_zoom = 10
pixel_to_mm_10x = 1.1    # 11 pixels = 10 mm
pixel_to_mm_25x = 2.7    # 27 pixels = 10 mm
deleted_circles = 0


start_corner_x = 0
start_corner_y = 0
end_corner_x = 0
end_corner_y = 0
circle = None
active_toplevel = None
create_circle_mode = False
did_drag = False

hough_circles = None
drawn_circles = None


def blur(img):
    blur_kernel_size = 5    # 5 x 5 filter
    return cv2.medianBlur(img, blur_kernel_size)

def basic_threshold(img):
    #gray = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def hough_transform():
    global blurred_img
    global min_radius
    global max_radius
    global min_distance_between_circles
    global found_circles
    if type(blurred_img) is np.ndarray:
        accumulator_to_image_ratio = 1.8
        canny_threshold = 30
        accumulator_threshold = 28
        found_circles = cv2.HoughCircles(image = blurred_img,
        			                     method = cv2.HOUGH_GRADIENT,
            			                 dp = accumulator_to_image_ratio,
                            			 minDist = min_distance_between_circles,
                                         param1 = canny_threshold,
                            			 param2 = accumulator_threshold,
                            			 minRadius = min_radius,
                            			 maxRadius = max_radius)
        found_circles = found_circles[0]
        # Convert circles to integer for processing
        if type(found_circles) is np.ndarray:
            found_circles = np.uint16(np.around(found_circles))
            return found_circles
        else:
            return False

def draw_circles(img, found_circles, highlight_index = -1):
    global radios
    global image_zoom
    global hough_circles
    global drawn_circles
    global hough_radius
    global drawn_radius
    radios = []
    hough_radius = []
    drawn_radius = []
    total_found_circles = 0
    if type(hough_circles) is np.ndarray:
        for circle_index, circle in enumerate(hough_circles):
            # draw the outer circle
            if circle_index == highlight_index:
                cv2.circle(img, (circle[0],circle[1]),circle[2],(0,0,255),3)
            else:
                cv2.circle(img,(circle[0],circle[1]),circle[2],(255,0,0),3)
            radius_in_pixels = circle[2]
            radius_in_mm = float(radius_in_pixels / pixel_to_mm_10x)
            if image_zoom == 25:
                radius_in_mm = float(radius_in_pixels / pixel_to_mm_25x)
            radios.append(radius_in_mm)
            hough_radius.append(radius_in_mm)
        total_found_circles = len(hough_circles)
    if type(drawn_circles) is np.ndarray:
        for circle_index, circle in enumerate(drawn_circles):
            # draw the outer circle
            if circle_index == highlight_index - total_found_circles:
                cv2.circle(img, (circle[0],circle[1]),circle[2],(0,0,255),3)
            else:
                cv2.circle(img,(circle[0],circle[1]),circle[2],(0,255,255),3)
            radius_in_pixels = circle[2]
            radius_in_mm = float(radius_in_pixels / pixel_to_mm_10x)
            if image_zoom == 25:
                radius_in_mm = float(radius_in_pixels / pixel_to_mm_25x)
            radios.append(radius_in_mm)
            drawn_radius.append(radius_in_mm)
        total_found_circles += len(drawn_circles)
    if len(radios) > 0:
        radius_label_text.set("Radio promedio: " + "{0:.3f}".format((float(sum(radios)/len(radios)))) + ' μm.')
        circle_label_text.set("Circulos encontrados: " + str(total_found_circles))
    else:
        radius_label_text.set("Radio promedio: 0 μm.")
        circle_label_text.set("Circulos encontrados: 0")
    return img

def save_found_circles_data(found_circles, window):
    global img_path
    global radios
    global image_zoom
    global SAVE_ERROR_DATA
    current_month_and_year = datetime.date.today().strftime("%m-%Y")
    final_img_path = os.getcwd() + '/Results/Processed/' + img_path.split('/')[-1]
    file_path = os.getcwd() + '/Results/Data/' + 'PROCESSED_' + current_month_and_year + '.csv'
    file_writing_mode = 'a'    # append mode (writes on existent file)
    found_circle_data = ''
    if not os.path.isfile(file_path):
        found_circle_data = 'image type;celulose concentration;zoom;original image path;found circles;average radius;date;final image path\n'
        file_writing_mode = 'w'   # write mode (creates a new file)
    todays_file = open(file_path, file_writing_mode)

    final_img_path = save_final_image()
    found_circle_data += ("PROCESSED;" + img_path.split('%')[0].split(' ')[1] + ";"
                            + str(image_zoom)
                            + ";" + img_path
                            + ";" + str(len(found_circles))
                            + ';' + str(sum(radios)/len(radios)) + ';'
                            + datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")
                            + ';' + final_img_path + '\n')
    todays_file.write(found_circle_data)
    todays_file.close()
    save_comparison_data()
    if SAVE_ERROR_DATA:
        save_error(found_circles)
    window.destroy()

def save_comparison_data():
    global img_path
    current_month_and_year = datetime.date.today().strftime("%m-%Y")
    native_data_path = os.getcwd() + '/Results/Data/' + 'NATIVE_' + current_month_and_year + '.csv'
    processed_data_path = os.getcwd() + '/Results/Data/' + 'PROCESSED_' + current_month_and_year + '.csv'
    file_path = os.getcwd() + '/Results/Data/' + 'COMPARISON_' + current_month_and_year + '.csv'

    processed_last_data = open(processed_data_path, 'r').readlines()[-1].strip().split(';')
    native_last_data = open(native_data_path, 'r').readlines()[-1].strip().split(';')

    file_writing_mode = 'a'    # append mode (writes on existent file)
    found_circle_data = ''
    if not os.path.isfile(file_path):
        found_circle_data = 'celulose concentration;zoom;original native image path;original processed image path;final native image path;final processed image path;found circles (native);found circles (processed);average radius (native);average radius (processed);date\n'
        file_writing_mode = 'w'   # write mode (creates a new file)
    todays_file = open(file_path, file_writing_mode)
    found_circle_data += (native_last_data[1] + ";"
                            + native_last_data[2] + ";"
                            + native_last_data[3] + ";" + processed_last_data[3]
                            + ";" + native_last_data[7] + ";" + processed_last_data[7]
                            + ";" + native_last_data[4] + ";" + processed_last_data[4]
                            + ";" + native_last_data[5] + ";" + processed_last_data[5]
                            + ";" + datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")
                            + '\n')
    todays_file.write(found_circle_data)
    todays_file.close()

def select_image():
    global img
    global img_path
    global image_zoom
    # open a file chooser dialog and allow the user to select an input
    # image
    img_path = tkFileDialog.askopenfilename()
    image_zoom = int(img_path.split('x')[1].split('_')[0].split('P')[0])
    # ensure a file path was selected
    if len(img_path) > 0:
        img = cv2.imread(img_path, img_color)
        min_distance_between_circles_slider.pack()
        min_radius_slider.pack()
        max_radius_slider.pack()
        circle_label.pack(pady=(15,0))
        radius_label.pack()
        use_hough_checkbutton.pack(pady=10, padx=10, anchor=E)
        next_btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        photo_btn.config(text = "Cambiar imagen")
        canvas.pack(expand=YES, fill=BOTH)
        process_image()

def reprocess_image():
    global img
    global min_radius
    global max_radius
    global min_distance_between_circles
    global final_image
    global hough_circles
    global found_circles
    global drawn_circles
    found_circles = drawn_circles
    hough_circles = None
    blurred_img = blur(img)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thresholded_img = basic_threshold(image)
    # Remove noise (small floating particles) by blurring
    blurred_img = blur(thresholded_img)
    if type(img) is np.ndarray:
        min_radius = min_radius_slider.get()
        max_radius = max_radius_slider.get()
        min_distance_between_circles = min_distance_between_circles_slider.get()
        # Hough Transform to find circles
        if is_hough_enabled.get():
            hough_circles = hough_transform()
            if type(drawn_circles) is np.ndarray:
                found_circles = np.concatenate((hough_circles, drawn_circles), axis=0)
            else:
                found_circles = hough_circles

        raw_img = cv2.imread(img_path, 1)
        # Draw the found circles on the original image
        img_with_circles = draw_circles(raw_img, found_circles)
        final_image = img_with_circles
        show_image(img_with_circles)


def process_image():
    global img
    global blurred_img
    global final_image
    global hough_circles
    global drawn_circles
    drawn_circles = None
    use_hough_checkbutton.select()
    # Apply createCLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thresholded_img = basic_threshold(image)
    # Apply blur to remove noise
    blurred_img = blur(thresholded_img)
    # Hough Transform to find circles
    found_circles = hough_transform()
    hough_circles = found_circles
    # Draw the found circles on the original image and save it
    raw_img = cv2.imread(img_path, 1)
    img_with_circles = draw_circles(raw_img, found_circles)
    final_image = img_with_circles
    show_image(img_with_circles)


def save_final_image():
    global final_image
    global img_path
    final_img_path = os.getcwd() + '/Results/Processed/' + img_path.split('/')[-1]
    cv2.imwrite(final_img_path, final_image)
    return final_img_path


def show_image(img):
    global canvas
    global resize_x
    global resize_y
    global original_image_x
    global original_image_y
    global circle
    global image_on_canvas
    global img_path
    global image
    # SHOW IN GUI
    # Convert to COLOR_BGR
    image = img
    #image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert the images to PIL format...
    image = Image.fromarray(image)
    original_image_x, original_image_y = image.size
    image = image.resize((resize_x, resize_y),Image.ANTIALIAS)
    # ...and then to ImageTk format
    image = ImageTk.PhotoImage(image)
    # if the panels are None, initialize them
    canvas.itemconfigure(image_on_canvas, image = image)

def convert_point(coordenate_x, coordenate_y):
    global original_image_x
    global original_image_y
    global resize_x
    global resize_y
    x_ratio = float(original_image_x)/resize_x
    y_ratio = float(original_image_y)/resize_y
    return coordenate_x * x_ratio, coordenate_y * y_ratio

def center_window(toplevel):
    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))


def button_clicked(click_event):
    global found_circles
    global is_circle_selected
    global active_toplevel
    global create_circle_mode
    x, y  = convert_point(click_event.x, click_event.y)
    found_index = -1
    raw_img = cv2.imread(img_path, 1)
    if not create_circle_mode:
        for circle_index, circle in enumerate(found_circles):
            center_x = circle[0]
            center_y = circle[1]
            radius = circle[2]
            if ((x - center_x)**2 + (y - center_y)**2) < radius**2:
                is_circle_selected = True
                create_circle = False
                found_index = circle_index
                img_with_circles = draw_circles(raw_img, found_circles, found_index)
                show_image(img_with_circles)
                if active_toplevel != None:
                    active_toplevel.destroy()
                toplevel = Toplevel()
                active_toplevel = toplevel
                delete_circle_btn = Button(toplevel, text="Eliminar Circulo", command=lambda : delete_circle(circle_index, toplevel))
                delete_circle_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
                center_window(toplevel)
                break
    else:
        on_start(click_event)

def delete_circle(index, toplevel):
    global found_circles
    global final_image
    global hough_circles
    global drawn_circles
    global deleted_circles
    raw_img = cv2.imread(img_path, 1)
    new_circles = np.array(np.delete(found_circles, index, axis=0))
    found_circles = new_circles
    if is_hough_enabled.get():
        if index < len(hough_circles):  # its a hough circle
            hough_circles = np.array(np.delete(hough_circles, index, axis=0))
            deleted_circles += 1
        else:
            drawn_circles = np.array(np.delete(drawn_circles, index - len(hough_circles), axis=0))
    else:
        drawn_circles = np.array(np.delete(drawn_circles, index, axis=0))
    img_with_circles = draw_circles(raw_img, found_circles)
    final_image = img_with_circles
    show_image(img_with_circles)
    is_circle_selected = False
    toplevel.destroy()

def callback(event):
    draw(event.x, event.y)

def draw(x, y):
    global start_corner_x
    global start_corner_y
    canvas.coords(circle, start_corner_x, start_corner_y, x, y)

def on_start(event):
    global start_corner_x
    global start_corner_y
    global end_corner_x
    global end_corner_y
    canvas.configure(cursor="hand1")
    # you could use this method to create a floating window
    # that represents what is being dragged.
    start_corner_x = event.x
    start_corner_y = event.y
    end_corner_x = start_corner_x
    end_corner_y = end_corner_y

def on_drag(event):
    global end_corner_x
    global end_corner_y
    global start_corner_x
    global start_corner_y
    global create_circle_mode
    global did_drag
    if create_circle_mode:
        did_drag = True
        end_corner_x = event.x
        end_corner_y = event.y
        dif_x = (start_corner_x - end_corner_x)
        dif_y = (start_corner_y - end_corner_y)
        if abs(dif_x) < abs(dif_y): # hay que dejar dif_y = dif_x
            if dif_y < 0:
                end_corner_y = end_corner_y + (abs(dif_x) - abs(dif_y))
            else:
                end_corner_y = end_corner_y - (abs(dif_x) - abs(dif_y))
        else:   # hay que dejar dif_x = dif_y
            if dif_x < 0:
                end_corner_x = end_corner_x + (abs(dif_y) - abs(dif_x))
            else:
                end_corner_x = end_corner_x - (abs(dif_y) - abs(dif_x))
        draw(end_corner_x, end_corner_y)

def on_drop(event):
    global start_corner_x
    global start_corner_y
    global end_corner_x
    global end_corner_y
    global found_circles
    global create_circle_mode
    global did_drag
    global drawn_circles
    global final_image
    if create_circle_mode and did_drag:
        # find the widget under the cursor
        center_x = start_corner_x + (end_corner_x - start_corner_x)/2
        center_y = start_corner_y + (end_corner_y - start_corner_y)/2
        radius = int(np.sqrt((start_corner_x-end_corner_x)**2+(start_corner_y-end_corner_y)**2)/(2*np.sqrt(2)))
        radius = int(radius*1.65)
        center_x, center_y = convert_point(center_x, center_y)
        center_x = int(center_x)
        center_y = int(center_y)
        if type(drawn_circles) is np.ndarray:
            drawn_circles = np.concatenate((drawn_circles, np.array([[center_x, center_y, radius]])), axis=0)
        else:
            drawn_circles = np.array([[center_x, center_y, radius]])
        if is_hough_enabled.get():
            found_circles = np.concatenate((found_circles, np.array([[center_x, center_y, radius]])), axis=0)
        else:
            found_circles = drawn_circles
        raw_img = cv2.imread(img_path, 1)
        img_with_circles = draw_circles(raw_img, found_circles)
        final_image = img_with_circles
        show_image(img_with_circles)
        start_corner_x = 0
        end_corner_x = 0
        start_corner_y = 0
        end_corner_y = 0
        draw(0,0)
        did_drag = False
        canvas.configure(cursor="arrow")

def on_control_press(event):
    global create_circle_mode
    create_circle_mode = True

def on_control_release(event):
    global create_circle_mode
    create_circle_mode = False

def save_error(found_circles):
    global img_path
    global radios
    global image_zoom
    global hough_circles
    global drawn_circles
    global hough_radius
    global drawn_radius
    current_month_and_year = datetime.date.today().strftime("%m-%Y")
    file_path = os.getcwd() + '/Results/Data/' + 'PROCESSED_ERROR_' + current_month_and_year + '.csv'
    file_writing_mode = 'a'    # append mode (writes on existent file)
    found_circle_data = ''
    if not os.path.isfile(file_path):
        found_circle_data = 'image type;celulose concentration;zoom;total found circles;hough circles;drawn circles;total radius;hough radius;drawn radius;total avg radius;hough avg radius;drawn avg radius;deleted circles;date\n'
        file_writing_mode = 'w'   # write mode (creates a new file)
    todays_file = open(file_path, file_writing_mode)
    found_circle_data += ("PROCESSED;" + img_path.split('%')[0].split(' ')[1]
                            + ';' + str(image_zoom)
                            + ';' + str(get_array_len(found_circles))
                            + ';' + str(get_array_len(hough_circles))
                            + ';' + str(get_array_len(drawn_circles))
                            + ';' + json.dumps(make_histogram(radios))
                            + ';' + json.dumps(make_histogram(hough_radius))
                            + ';' + json.dumps(make_histogram(drawn_radius))
                            + ';' + str(get_average_radius(radios))
                            + ';' + str(get_average_radius(hough_radius))
                            + ';' + str(get_average_radius(drawn_radius))
                            + ';' + str(deleted_circles)
                            + ';' + datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S") + '\n')
    todays_file.write(found_circle_data)
    todays_file.close()

def get_average_radius(arr):
    if not arr:
        return 0
    else:
        return sum(arr)/len(arr)

def get_array_len(arr):
    if type(arr) is not np.ndarray:
        return 0
    else:
        return len(arr)

def make_histogram(radius):
    radius_dictionary = dict()
    for r in radius:
        r_int = int(r)
        if r_int in radius_dictionary:
            radius_dictionary[r_int] += 1
        else:
            radius_dictionary[r_int] = 1
    return radius_dictionary

if __name__ == '__main__':
    global canvas
    global image_on_canvas
    global root

    root = Tk()
    root.title("Reconocimiento Muestra Procesada")

    min_distance_var = DoubleVar()
    min_distance_between_circles_slider = Scale( root, length=200, from_=1, to=250, label="Distancia entre Centros (px):", variable = min_distance_var, orient="horizontal", command= lambda x: reprocess_image())
    min_distance_between_circles_slider.set(180)
    min_radius_var = DoubleVar()
    min_radius_slider = Scale( root,  length=200, from_=1, to=250, label="Radio Minimo (px):", variable = min_radius_var, orient="horizontal", command=lambda x: reprocess_image() )
    min_radius_slider.set(75)
    max_radius_var = DoubleVar()
    max_radius_slider = Scale( root,  length=200, from_=1, to=250, label="Radio Máximo (px):", variable = max_radius_var, orient="horizontal", command=lambda x: reprocess_image() )
    max_radius_slider.set(90)
    circle_label_text = StringVar()
    circle_label = Label(root, textvariable=circle_label_text)
    radius_label_text = StringVar()
    radius_label = Label(root, textvariable=radius_label_text)

    is_hough_enabled = BooleanVar()
    use_hough_checkbutton = Checkbutton(root, text="Hough Circles", variable=is_hough_enabled, onvalue=True, offvalue=False, command=lambda : reprocess_image() )
    use_hough_checkbutton.select()

    canvas = Canvas(root, width=resize_x, height=resize_y)
    #canvas.pack(expand=YES, fill=BOTH)
    temp_img = ImageTk.PhotoImage(file='hist.png')
    image_on_canvas = canvas.create_image(0, 0, image=temp_img, anchor=NW)
    canvas.bind("<ButtonPress-1>", button_clicked)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_drop)
    canvas.focus_set()
    canvas.bind("<KeyPress-0xffe3>", on_control_press)
    canvas.bind("<KeyRelease-0xffe3>", on_control_release)

    #canvas.bind("<Button 1>",select_circle)
    #canvas.pack()
    circle = canvas.create_oval(0, 0, 0, 0, outline='yellow')
    #canvas = Label(image=image)
    #canvas.pack()

    photo_btn = Button(root, text="Seleccionar imagen", command= lambda : select_image(), width = 80)
    photo_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    next_btn = Button(root, text="Siguiente", command= lambda : save_found_circles_data(found_circles, root))

    root.mainloop()
