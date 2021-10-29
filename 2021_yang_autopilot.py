import numpy as np
import time
import tensorflow as tf
from robotPi import robotPi
import cv2
import os
from rev_cam import rev_cam  # 图片翻转


width = 480
height = 180
channel = 1
inference_path = tf.Graph()
filepath = os.getcwd() + '/model/auto_drive_model/old/-339'
temp_image = np.zeros(width * height * channel, 'uint8')


def auto_pilot():
    cap = cv2.VideoCapture("/dev/video0")
    with tf.Session(graph=inference_path) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.import_meta_graph(filepath + '.meta')
        saver.restore(sess, filepath)

        tf_X = sess.graph.get_tensor_by_name('input:0')
        pred = sess.graph.get_operation_by_name('pred')
        number = pred.outputs[0]
        prediction = tf.argmax(number, 1)
        
        frame_num = 0
        zhong_dian = 0
        robot.movement.start_hit()
        while cap.isOpened():
            ret, frame = cap.read()
            frame = rev_cam(frame)
            resized_height = int(width * 0.75)
            # 计算缩放比例
            frame = cv2.resize(frame, (width, resized_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 选取图像的下半部分 [480*0.75-180]*480（高*宽）
            res = frame[resized_height - height:, :]
            #cv2.imshow("frame", res)
            # 二值化
            ret,thresh1 = cv2.threshold(res,95,255,cv2.THRESH_BINARY)            
            #kernel = np.ones((12, 12), dtype=np.uint8)
            #closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("thresh", thresh1) 
            cv2.moveWindow("thresh",700,0)            
            cv2.waitKey(1)
            frame = np.array(thresh1, dtype=np.float32)
            value = prediction.eval(feed_dict={tf_X: np.reshape(frame, [-1, height, width, channel])})            
            print('img_out:', value)
            
            if value == 0:
                print("forward")
                frame_num += 1               
                robot.movement.move_forward(speed=yby_speed,times=yby_delay)

            elif value == 1:
                print("left")
                frame_num += 1                
                if (time.time() - start) <= yby_time:
                    robot.movement.turn_left(speed=5, times=300)
                    #robot.movement.left_ward()
                else:    
                    robot.movement.turn_left(speed=10, times=50)
                    time.sleep(0.2)

            elif value == 2:
                print("right")
                frame_num += 1                
                if (time.time() - start) <= yby_time:
                    robot.movement.turn_right(speed=5, times=300)
                    #robot.movement.right_ward()
                else:
                    robot.movement.turn_right(speed=10, times=50)
                    time.sleep(0.2)
                    
            elif value == 3:
                frame_num += 1                
                if (time.time() - start) <= yby_time:  
                    robot.movement.move_forward(speed=yby_speed,times=yby_delay)                   
                else:
                    zhong_dian = zhong_dian + 1
                    if(zhong_dian == 2):
                        print("stop sign")
                        print(frame_num,"Frame")
                        robot.movement.move_forward(speed=30,times=350)
                        time.sleep(0.5)
                        cap.release()
                        cv2.destroyAllWindows()
                    
            elif value == 4:
                print("Banner forward")
                frame_num += 1
                robot.movement.move_forward(speed=yby_speed,times=yby_delay)
                    
            elif value == 5:
                print("Banner left")
                frame_num += 1               
                robot.movement.left_ward()
                #robot.movement.turn_left(speed=5, times=300)

            elif value == 6:
                print("Banner right")
                frame_num += 1                
                robot.movement.right_ward()
                #robot.movement.turn_right(speed=5, times=300)
            
            elif cv2.waitKey(1) & 0xFF ==ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

def ding_wei():
    cap = cv2.VideoCapture("/dev/video1")
    time.sleep(0.5)
    while(1):
        ret, frame = cap.read()
        frame = rev_cam(frame)
        # 重设图像尺寸
        frame_old = cv2.resize(frame, (480,360))
        # RGB转HSV
        frame_hsv1 = cv2.cvtColor(frame_old,cv2.COLOR_BGR2HSV)
        frame_hsv2 = cv2.cvtColor(frame_hsv1,cv2.COLOR_BGR2HSV)
        # 红色范围
        low_range = np.array([0, 0, 208])
        high_range = np.array([177, 166, 255])
        frame = cv2.inRange(frame_hsv2, low_range, high_range)
        # 图像腐蚀和扩张 
        kernel_1 = np.ones((5,5),np.uint8)
        kernel_2 = np.ones((12,12),np.uint8)
        frame = cv2.erode(frame,kernel_1)        #腐蚀图像
        frame = cv2.dilate(frame,kernel_2)
        # 查找符合的轮廓 
        _,contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            target = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(target)
            print(int(x),int(radius))
            # 左右调整
            if x > 320:
                robot.movement.move_right(speed=10, times=50)
            elif x < 250:
                robot.movement.move_left(speed=10, times=50)
            else:
                robot.movement.move_forward(speed=10, times=100)                
                if radius >= 46:
                    robot.movement.start_hit()
                    robot.movement.hit()
                    time.sleep(1)
                    robot.movement.start_hit()
                    cv2.destroyAllWindows()
                    cap.release()
                    break                
            
        cv2.namedWindow("old",cv2.WINDOW_NORMAL)
        #cv2.namedWindow("hsv1",cv2.WINDOW_NORMAL)
        #cv2.namedWindow("hsv2",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Y",cv2.WINDOW_NORMAL)       #调整窗口大小及位置
        cv2.imshow("old", frame_old)
        cv2.moveWindow("old",700,0)
        #cv2.imshow("hsv1", frame_hsv1)
        #cv2.imshow("hsv2", frame_hsv2)
        cv2.imshow("Y", frame)
        cv2.moveWindow("Y",700,280)
        k = cv2.waitKey(1)
        if k & 0xFF ==ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':  
    yby_time = 40
    yby_speed = 33
    yby_delay = 155
    robot = robotPi()
    start = time.time()
    auto_pilot()
    ding_wei()
    end = time.time()
    print(end-start,"S")
    #print(frame_number/(end-start),"f/s")
