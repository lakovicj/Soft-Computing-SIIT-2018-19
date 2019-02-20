import numpy as np
import cv2 # OpenCV

def process_videos():

    file = open("out.txt", "w")
    if not file:
        print("Somethings wrong with file writer!")
        return -1
    file.write("SW51/2015, Jovan Lakovic\n")
    file.write("file,count\n")


    for vid_index in range(1,11):
        people_in_video = 0
        while True:
            vid_name = "data/video" + str(vid_index) + ".mp4"
            capture = cv2.VideoCapture(vid_name)
            frame_cnt = 0
            if not capture.isOpened():
                print("Something's off! Error while opening video")
                return -1

            # kroz frejmove
            while capture.get(1) < capture.get(7) - 1:
                frame_cnt += 1

                ret_val, frame1 = capture.read()
                ret_val, frame2 = capture.read()

                # pronadji plato i izdvoji ga ... ovo samo za prvi frejm
                if frame_cnt == 1:
                    inverted_frame1 = 255 - frame1
                    gray_frame1 = cv2.cvtColor(inverted_frame1, cv2.COLOR_BGR2GRAY)
                    _, binary_frame1 = cv2.threshold(gray_frame1, 190, 255, cv2.THRESH_BINARY)
                    eroded_frame1 = cv2.erode(binary_frame1, np.ones((3,3)), iterations=1)

                    # izdvoj sve konture
                    _, contours, plato_hierarchy = cv2.findContours(eroded_frame1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    show_plato = frame1.copy()

                    # izdvoj samo plato -> max kontura
                    only_plato = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(only_plato)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # iskoriguj granice platoa malo
                    bottom_left = box[0]
                    top_left = box[1]
                    top_left[0] += 40
                    top_right = box[2]
                    top_right[0] -= 45
                    bottom_right = box[3]


                    #real_box = np.array([[165, 467], [180, 114], [470, 91], [535, 444]])
                    real_box = np.array([bottom_left, top_left, top_right, bottom_right])
                    #cv2.drawContours(show_plato, [real_box], 0, (0, 0, 255), 2)
                    #cv2.imshow("samo plato", show_plato)

                    # segment za prebrojavanje -> na osnovu platoa
                    segment_bottom_left = [top_left[0], top_left[1] + 100]
                    segment_top_left = [top_left[0], top_left[1] + 80]
                    segment_top_right = [top_right[0], top_right[1] + 80]
                    segment_bottom_right = [top_right[0], top_right[1] + 100]
                    segment_box = np.array([segment_bottom_left, segment_top_left, segment_top_right, segment_bottom_right])
                    cv2.drawContours(show_plato, [segment_box], 0, (0, 0, 255), 2)
                    #cv2.imshow("plato ", show_plato)

                # background substraction
                # ipak losi rez tkd nista od ovoga na kraju..............................
                gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                difference_img = cv2.absdiff(gray_frame1, gray_frame2)
                threshold_img = cv2.adaptiveThreshold(difference_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)


                # na svakih N frejmova ide detekcija objekata tj pesaka
                if frame_cnt%7 == 0:
                    # za detektovane pesake
                    detected_list = []
                    # kropuj samo plato.. ipak samo segment platoa kako se ne bi obradjivala cela slika
                    # i posle proveravale granice, ovako samo obradjujes ovo
                    # background substraction mi ni ne treba na kraju zapravo, zbog segmenta
                    detection_segment = frame1[segment_top_left[1]:segment_bottom_left[1]+10,segment_bottom_left[0] + 60:segment_bottom_right[0]-70]

                    # samo za prikaz detektovanih posle
                    detected_img = detection_segment.copy()

                    gray_segment = cv2.cvtColor(detection_segment, cv2.COLOR_BGR2GRAY)
                    gray_segment = 255 - gray_segment

                    #cv2.imshow("gray_segment", gray_segment)

                    # kao params za blockSize i C sa prvih vezbi -> najveca tacnost, ostali pokusaji svi <80%
                    threshold_segment = cv2.adaptiveThreshold(gray_segment, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

                    # closing (dilate pa onda erode)
                    close_segment = closing(threshold_segment)

                    #cv2.imshow("close-segment", close_segment)

                    # detektuj pesake

                    _, contours, hierarchy = cv2.findContours(close_segment, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        #print(cnt.shape)
                        (x, y), radius = cv2.minEnclosingCircle(cnt)

                        radius = int(radius)
                        if 1 < radius < 8:
                            detected_list.append(cnt)
                            people_in_video += 1


                    cv2.drawContours(detected_img, detected_list, -1, (0, 255, 0), 1)
                    #cv2.imshow("detektovani", detected_img)
                    cv2.waitKey(50)



            #cv2.waitKey(50)
            cv2.destroyAllWindows()
            #capture.release()
            break

        print("video number: " + str(vid_index))
        print("num of people: " + str(people_in_video))
        file.write("video" + str(vid_index) + ".mp4," + str(people_in_video) + '\n')
    file.close()


def erode(bin_image):
    # MORPH.RECT i ostale sve su mi davale manju tacnost od obicne 3x3
    k = np.ones((3,3))
    eroded = cv2.erode(bin_image, k, iterations=2)
    return eroded

def dilate(bin_image):
    k = np.ones((3,3))
    dilated = cv2.dilate(bin_image, k, iterations=1)
    return dilated


def opening(bin_image):
    temp = erode(bin_image)
    opened = dilate(temp)
    return opened

def closing(bin_image):
    temp = dilate(bin_image)
    closed = erode(temp)
    return closed


def hist(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)

    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1

    return (x, y)




if __name__ == '__main__':

    process_videos()