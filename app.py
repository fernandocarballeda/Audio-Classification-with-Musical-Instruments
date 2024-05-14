import load_model_weights
import load_model

# ===================================================================
# Functions
# ===================================================================

def model_predict():

    classes = ['Acoustic_guitar', 'Bass_drum', 'Cello', 'Clarinet', 'Double_bass', 'Flute', 'Hi_hat', 'Piano', 'Saxophone', 'Snare_drum', 'Violin_of_fiddle']
    model_weights = load_model_weights('models/lstm.h5')
    my_trained_model = load_model('models/lstm.h5')
    my_trained_model.load_state_dict(model_weights)
    predicted_label = predict(my_trained_model, aud)
    return classes[predicted_label]

def predict(my_trained_model, aud):

    tensor = aud
    my_trained_model.eval()
    with torch.no_grad():
        output = my_trained_model(tensor.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    return predicted

# ========================================================================
# Application Config
# ========================================================================
st.set_page_config(layout="wide", page_title="Musical instruments prediction")
col01, col02, col03 = st.columns(3)
with col01:
    st.write('# Musical instruments prediction')

st.write("### By: Fernando Carballeda & Lucas Justo")


col21, col22, col23 = st.columns(3)
with col21:
    st.write("#### Drag and Drop Audio")
    uploaded_audio = st.file_uploader("Choose an audio...", type=["mp3"])
    if uploaded_audio is not None:
        img = Image.open(uploaded_image)
        fig, ax = plt.subplots(figsize=(12, 6))
        img = transform_image(img)
        plt.imshow(img["img_tensor"].squeeze(), cmap='gray')
        plt.colorbar()
        st.pyplot(fig)
    

col41, col42, col43 = st.columns(3)
with col41:
    if st.button("Predict"):
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            predicted_class = model_predict()  # Llama a la función de predicción
            st.write("##### Predicted Instrument:", predicted_class)