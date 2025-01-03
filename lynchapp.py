import streamlit as st
from Transformer import PositionalEncoding, MultiHeadAttentionLayer, predict
import tensorflow as tf

filename = "model.h5"
model = tf.keras.models.load_model(
    filename,
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)

# Streamlit UI
st.image("background.png", use_container_width=True)

st.markdown(
    """
    <h1 style="color: darkgreen; text-align: center; font-size: 45px; font-weight: bold;">
        I am Peter Lynch! Ask me anything
    </h1>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_input("Your question:")
if user_input:
    st.markdown(f"**You asked:** {user_input}")
    try:
        # Use the predict_answer function from Transformer.py
        answer = predict(user_input)
        st.markdown(f"**Peter Lynch says:** {answer}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
