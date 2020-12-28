import streamlit as st
import main
import tempfile
import cv2 as cv


def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

load_css('style.css')


with st.sidebar:
    t = "<span class='bold blue highlight'>Chọn hình ảnh backgound</span>"
    st.markdown(t, unsafe_allow_html=True)
    b = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if b:
        bg = tempfile.NamedTemporaryFile(delete=False)
        bg.write(b.read())
        bg.close()



    t = "<span class='bold red highlight'>Chọn hình ảnh hoặc video</span>"
    st.markdown(t, unsafe_allow_html=True)
    type_fg_file = st.selectbox("", ['video', 'image'])

    if type_fg_file  == 'video':
        text = 'video'
        type = 'mp4'
    else:
        text = 'image'
        type = ['png', 'jpg', 'jpeg']


    f = st.file_uploader('', type, key='12')
    btn = st.button('Run')
    if not btn:
        st.stop()
    fg = tempfile.NamedTemporaryFile(delete=False)
    fg.write(f.read())
    fg.close()



if text == 'video':

    bg = cv.imread(bg.name)
    bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)

    coef = main.load_model()
    vf = cv.VideoCapture(fg.name)

    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        processed_frame = main.remove_green(frame, coef, bg)

        stframe.image(processed_frame, width=682)

else:
    image = main.process(fg.name, bg.name, 'image')
    st.image(image, width=682)