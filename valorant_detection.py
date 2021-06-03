import time
import mss
import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import pyautogui
from multiprocessing import Pipe
from multiprocessing import Process

# Initialize monitor size
maxWidth = 640
maxHeight = 480


labels_path = 'training/label_map.pbtxt'

# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_pos = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def shoot(mid_x, mid_y):
  x = mid_x*maxWidth
  y = mid_y*maxHeight
  pyautogui.moveTo(int(x),int(y))
  pyautogui.click()

def grab_screen(p_input):
  sct = mss.mss()
  monitor = {"top": 160, "left": 160, "width": maxWidth, "height": maxHeight}
  
  while True:
    #Grab screen image
    img = np.array(sct.grab(monitor))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Put image from pipe
    p_input.send(img)
 
def add_player(img, mid_x, mid_y, player_list,rgb):
  player_list.append([mid_x,mid_y])
  cv2.circle(img,(int(mid_x*maxWidth),int(mid_y*maxHeight)), 3, rgb, -1)
  
def player_detection(pipe_out, pipe_in2):
  # Detection
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      while True:

        img = pipe_out.recv()
        # expand to [1, None, None, 3]
        img_new_dim = np.expand_dims(img, axis=0)

        # Actual detection.
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Visualization of the results of a detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_new_dim})
        vis_util.visualize_boxes_and_labels_on_image_array(
            pipe_out.recv(),
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_pos,
            use_normalized_coordinates=True,
            line_thickness=3)

        # Send detection image to pipe2
        pipe_in2.send(img)

        enemy_head_locs = []
        enemy_locs = []
        team_locs = []

        for i in range(len(boxes[0])):
          if scores[0][i] >= 0.5:
            player_list = None
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            rgb = (50,150,255)
            curr_class = classes[0][i]

            if curr_class ==1:
              player_list = enemy_locs
            elif curr_class == 2:
              player_list = enemy_head_locs
              rgb = (0,0,255)
            elif curr_class == 3:
              player_list = team_locs
     
            add_player(img,mid_x,mid_y,player_list,rgb)
              
          if len(enemy_head_locs) > 0:
            shoot(enemy_head_locs[0][0], enemy_head_locs[0][1])
          elif len(enemy_locs) > 0:
            shoot(enemy_locs[0][0], enemy_locs[0][1])


def display_img(p_output2):
  fps = start_time = 0 
  MAX_TIME = 1 
  while True:
    image_np = p_output2.recv()
    # Show image with detection
    cv2.imshow('FPS', image_np)
    # Bellow we calculate our FPS
    fps+=1
    curr_time = time.time() - start_time
    if curr_time >= MAX_TIME :
      print("FPS: ", fps / curr_time)
      fps = 0
      start_time = time.time()

    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
      cv2.destroyAllWindows()
      break

  pipe_in1 = Pipe()[1]
  pipe_out2, pipe_in2 = Pipe()

  proc1 = Process(target=grab_screen, args=(pipe_in1))
  proc2 = Process(target=player_detection, args=(pipe_out2,pipe_in2))
  proc3 = Process(target=display_img, args=(pipe_out2))

  processes = [proc1,proc2,proc3]

  for proc in processes:
    proc.start()


