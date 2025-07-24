import streamlit as st
st.set_page_config(layout="wide", page_title="Xplainit", page_icon="https://cdn-icons-png.flaticon.com/512/4712/4712032.png")
# 🤖
st.markdown("""
    <div class="chatbot-button">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712032.png" alt="Chatbot Icon"/>
    </div>
""", unsafe_allow_html=True)


import cvzone
import cv2 #Video Capture
from cvzone.HandTrackingModule import HandDetector #Hand detector
import numpy as np #Manage Pixel data as array
import google.generativeai as genai #Gemini AI
from PIL import Image #Capture and send image

import threading #Avoid screen freeze and to synchronize

def speak_text(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

import pyttsx3
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


# UI Sidebar Controls
st.sidebar.title("Controls")
brush_size = st.sidebar.slider("🖌 Brush Size", 1, 50, 10)
brush_color = st.sidebar.color_picker("Brush Color", "#FF00FF")
clear_canvas_btn = st.sidebar.button("Clear Drawing")
speak_answer = st.sidebar.checkbox("🔊 Speak Answer", value=True)

# Chatbot
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Math Chatbot")
chat_input = st.sidebar.text_input("Ask a math question:")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if chat_input:
    try:
        with st.spinner("💬 Thinking..."):
            chat_response = genai.GenerativeModel('gemini-1.5-flash').generate_content(chat_input).text
            st.session_state.chat_history.append(("You", chat_input))
            st.session_state.chat_history.append(("Bot", chat_response))
    except Exception as e:
        chat_response = f"Error: {e}"
        st.session_state.chat_history.append(("Bot", chat_response))

# Display chat history
for sender, msg in st.session_state.chat_history[::-1]:  # latest on top
    if sender == "You":
        st.sidebar.markdown(f"**🧍‍♂️ {sender}:** {msg}")
    else:
        st.sidebar.markdown(f"**🤖 {sender}:** {msg}")

st.title('Xplainit- Your AI Math Tutor')
st.subheader('Think it, Draw it, Xplainit')

col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('📹 Activate Webcam', value=True)
    FRAME_WINDOW = st.empty()

with col2:
    st.subheader("📸 Handwritten Math Answer")
    answer_box = st.empty()

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')



# OpenCV Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1,
                        detectionCon=0.7, minTrackCon=0.5)

prev_pos = None
canvas = None
output_text = ""
tts_triggered = False # Track if TTS has been triggered

st.subheader("🤚 How to Use")
st.markdown("""
- **✌️ Draw**  
  Hold up **index + middle** fingers. Move your hand in front of the camera to draw on the canvas.
- **🖐️ Solve**  
  Hold up **all 5 fingers**. The app will capture your drawing and ask Gemini to solve it.
- **✊ Clear**  
  Make a **fist** (all fingers down). This will erase the canvas so you can start a new problem.
""")

# Webcam Drawing Loop
while run:
    success, img = cap.read()
    if not success:
        st.error("❌ Failed to access camera.")
        break

    img = cv2.flip(img, 1)

    # Create or clear canvas
    if canvas is None or clear_canvas_btn:
        canvas = np.zeros_like(img)
        clear_canvas_btn = False
        output_text = ""
        tts_triggered = False

    # Detect hand & gesture
    hands, _ = detector.findHands(img, draw=False, flipType=True)
    if hands:
        lmList = hands[0]["lmList"]
        fingers = detector.fingersUp(hands[0])

        # Draw on canvas if only index & middle fingers are up
        if fingers == [0, 1, 1, 0, 0]:
            curr = tuple(lmList[8][0:2])
            if prev_pos is None:
                prev_pos = curr
            cv2.line(canvas, curr, prev_pos,
                     tuple(int(brush_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
                     brush_size)
            prev_pos = curr
        else:
            prev_pos = None

        # AI solve: all 5 fingers up
        if fingers == [1, 1, 1, 1, 1] and output_text == "":
            with st.spinner('🤖 Solving your math problem...'):
                pil_img = Image.fromarray(canvas)
                try:
                    response = model.generate_content(["Solve this math problem:", pil_img])
                    output_text = response.text
                    answer_box.markdown(f"** Answer:** {output_text}")
                    #st.balloons()


                    tts_triggered = False  # Reset to allow speaking below
                except Exception as e:
                    output_text = f"Error: {e}"
                    tts_triggered = True  # Don’t speak errors

        # Clear screen gesture (✊)
        if fingers == [0, 0, 0, 0, 0]:
            canvas = np.zeros_like(img)
            prev_pos = None
            output_text = ""
            tts_triggered = False


    combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(combined, channels="BGR")

    if output_text:
        answer_box.markdown(f"** Answer:** {output_text}")
        if speak_answer and not tts_triggered:
            speak_text(output_text)
            tts_triggered = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
