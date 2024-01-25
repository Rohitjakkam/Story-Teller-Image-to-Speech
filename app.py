from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
import streamlit as st

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Load environment variables from a .env file
load_dotenv(find_dotenv())

def img2text(url):
    # Use the image-to-text pipeline
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    # Generate text from the image URL
    text = image_to_text(url)[0]['generated_text']
    print("Image Text:", text)
    return text

# Example usage

# Generate a story based on the image text


def generate_story(scenario):
    template = """
    You are a storyteller;
    You can generate a short story based on a simple narrative, the story should be no more than 30 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name='gpt-3.5-turbo', temperature=1), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    print("Generated Story:", story)
    return story

# Generate a story based on the scenario


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_conformer_fastspeech2"
    headers = {"Authorization": "Bearer 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"}
    payloads = {
        "inputs": message
    }

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    audio_bytes = query(payloads)

    # Save the audio as an MP3 file
    with open('audio.mp3', 'wb') as file:
        file.write(audio_bytes)

    print("Audio file saved as 'audio.mp3'")

    # You can also play the audio with IPython.display
    from IPython.display import Audio
    Audio(audio_bytes)

def main():

    st.set_page_config(page_title="Image to Audio Story",page_icon="💎")

    st.header("Turn image into audio story")
    uploaded_file = st.file_uploader("Choose an image from your device...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption="Uploaded Image.",
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        generated_story = generate_story(scenario)
        text2speech(generated_story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(generated_story)

        st.audio("audio.mp3")




if __name__ == '__main__':
    main()
