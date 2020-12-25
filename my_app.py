import streamlit as st
import main
import tempfile
import cv2 as cv





b = st.file_uploader("Hình ảnh back ground", type=["png", "jpg", "jpeg"])
if b:
    bg = tempfile.NamedTemporaryFile(delete=False)
    bg.write(b.read())
    bg.close()
type_fg_file = st.selectbox("", ['video', 'image'])

if type_fg_file  == 'video':
    text = 'video'
    type = 'mp4'
else:
    text = 'image'
    type = ['png', 'jpg', 'jpeg']


f = st.file_uploader(text, type)
btn = st.button('Run')
if not btn:
    st.stop()
fg = tempfile.NamedTemporaryFile(delete=False)
fg.write(f.read())
fg.close()



if text == 'video':

    bg = cv.imread(bg.name)

    coef = main.load_model('model (5).pickle')
    vf = cv.VideoCapture(fg.name)

    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        processed_frame = main.remove_green(frame, coef, bg)

        stframe.image(processed_frame, width=682)

else:
    image = main.process(fg.name, bg.name, 'image')
    st.image(image, width=682)