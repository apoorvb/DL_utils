import cv2

cap = cv2.VideoCapture('Path\to\file')

fps = cap.get(5)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi',fourcc,  fps, (frame_width, frame_height))

while(cap.isOpened()):	
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    if (cap.get(1) % 50 == 0):
      print('50 done.')
    frame = cv2.cvtCOLOR( frame,cv2.COLOR_BGR2GRAY)
    out.write(frame)
  else:
    break

cap.release()
out.release()
