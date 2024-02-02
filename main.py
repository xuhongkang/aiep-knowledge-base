import streamlit as st
import os, openai, io
from stKB import StreamlitKB

openai.api_key = os.getenv("OPENAI_API_KEY")
TEAM_DIR_URL = "https://drive.google.com/drive/folders/1uxp4nP6-3gBRVDtmxG54b9Z8H8st3UBR"

def main():
    st.set_page_config(page_title="AIEP Knowledge Base", page_icon="icon.png", layout="wide")
    kb = StreamlitKB()

    header_container, burnes_logo_container, ips_logo_container = st.columns([1, 1, 1])
    header_container.markdown(f'<a href="{TEAM_DIR_URL}" target="_blank"><button style="display: inline-block; background-color: #4CAF50; color: white; padding: 8px 16px; font-size: 16px; cursor: pointer; border: none; border-radius: 4px;">Vist Our Project Directory!</button></a>', unsafe_allow_html=True)
    header_container.header("AI-EP Knowledge Base")
    burnes_logo_container.image("burnes.png", width=460)
    ips_logo_container.image("ips.png", width=420)
    st.subheader('Use the Existing Model, Ask Away!')

    message_container, submit_question_container = st.columns([6, 1])
    message = message_container.text_area("Hi, this is the AIEP Spring 2024 Project Development Team. Feel free to ask any questions about our project's definition, timeline, recent meetings, motivations, and more!")
    submit_question_container.write('Ctrl+Enter')
    ask_button_clicked = submit_question_container.button("Ask", key='submit_question')
    if ask_button_clicked and message:
        result = kb.generate_response(message)
        st.info(result)
        
    st.subheader("Something Doesn't Feel Right? Help Improve the Model!")
    uploaded_file = st.file_uploader("Upload File", type=['pdf'])
    if uploaded_file is not None:
        progress_bar = st.progress(0)
        file_name = uploaded_file.name
        st.write(f"Extracting from file: {file_name}")
        bytes_data = uploaded_file.getvalue()
        kb.update_from_pdf(file_name, io.BytesIO(bytes_data), progress_bar.progress)
        st.write(f'File Loaded: {file_name}')
    st.subheader("OR")

    question_container, answer_container, submit_change_container = st.columns([2, 4, 1])
    question = question_container.text_area('(Optional) For This Question...')
    answer = answer_container.text_area("Here's the Right Answer!")
    submit_change_container.write("...and off it goes!")
    if submit_change_container.button("Submit", key='submit_change'):
        kb.update_embeddings(question, answer, priority=True)



if __name__ == '__main__':
    main()
