import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import os
from pydub import AudioSegment
import soundfile as sf
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Class list and input shape
class_list = {0: "cailuong", 1: "catru",2: "cheo"}
input_shape = (369, 496, 3)



# Predict functions
def predict_new30s(audio_dir, model_path, save_dir="/content/drive/MyDrive/test_images"):
    """
    Dự đoán thể loại nhạc từ các file audio dài 30 giây.
    Input:
    - audio_dir: List chứa đường dẫn tới các file audio (.wav)
    - model: Mô hình dùng để dự đoán
    - model_path: Đường dẫn tới mô hình đã lưu.
    - save_dir: Thư mục lưu hình ảnh Mel Spectrogram
    Output:
    - y_pred: Danh sách chỉ số dự đoán của các audio
    - y_class: Danh sách nhãn dự đoán của các audio
    """

    # Load model
    model = load_model(model_path)

    y_pred = []
    y_class = []

    for dir in audio_dir:
        # Bước 1: Load file âm thanh
        y, sr = lb.load(dir, sr=None)
        
        # Bước 2: Tính Mel Spectrogram
        D = lb.stft(y, hop_length=256)
        S = np.abs(D)
        mel_spec = lb.feature.melspectrogram(S=S**2, sr=sr, hop_length=256, n_mels=512+64)
        mel_spec_dB = lb.power_to_db(mel_spec, ref=np.max)

        # Bước 3: Tạo thư mục lưu nếu chưa tồn tại
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Bước 4: Lưu hình ảnh Mel Spectrogram
        audio_file_name = os.path.basename(dir).replace(".wav", "")
        saved_img_root = os.path.join(save_dir, f"{audio_file_name}.png")
        fig, ax = plt.subplots()
        img = lb.display.specshow(mel_spec_dB, x_axis='time', y_axis='mel', ax=ax)
        ax.set_axis_off()
        plt.savefig(saved_img_root, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Bước 5: Load hình ảnh và tiền xử lý
        image = load_img(saved_img_root, target_size=(input_shape[0], input_shape[1], 3))
        image_array = img_to_array(image) / 255
        input_data = tf.expand_dims(image_array, 0)

        # Bước 6: Dự đoán
        pred = model.predict(input_data, verbose=0)
        pred_index = np.argmax(np.squeeze(pred))

        y_pred.append(pred_index)
        y_class.append(class_list[pred_index])

    return y_pred, y_class

def mp3_2_wav(dir, dst, sample_rate=22050):
    """
    Chuyển đổi file MP3 sang WAV.
    Input:
    - dir: Đường dẫn file MP3.
    - dst: Đường dẫn lưu file WAV.
    """
    sound = AudioSegment.from_mp3(dir)
    sound.set_frame_rate(sample_rate)
    sound.export(dst, format="wav")

def predict_new(audio_dir, model_path, unit_length=661500, save_dir="/content/drive/MyDrive/test_images"):
    """
    Dự đoán thể loại nhạc từ các file audio có độ dài bất kỳ.
    Mỗi file audio sẽ được chia thành các đoạn có độ dài `unit_length`, sau đó đưa vào mô hình để dự đoán.
    Kết quả dự đoán được lấy bằng phương pháp voting.

    Input:
    - audio_dir: List chứa đường dẫn các file audio (.wav hoặc .mp3) cần dự đoán.
    - model_path: Đường dẫn tới mô hình đã lưu.
    - unit_length: Độ dài mỗi đoạn audio sau khi chia (mặc định: 661500).
    - save_dir: Thư mục lưu ảnh Mel Spectrogram.
    Output:
    - y_pred_index: List chứa chỉ số dự đoán cho từng file audio.
    - y_pred_class: List chứa nhãn dự đoán tương ứng.
    """

    # Danh sách kết quả dự đoán
    y_pred_index = []
    y_pred_class = []

    for dir in audio_dir:
        # Chuyển đổi MP3 sang WAV nếu cần
        if dir.endswith(".mp3"):
            wav_dir = os.path.join(save_dir, os.path.basename(dir).replace(".mp3", ".wav"))
            mp3_2_wav(dir, wav_dir)
            dir = wav_dir

        # Load file audio
        audio, sr = lb.load(dir, sr=None)
        if len(audio) < unit_length:
            raise ValueError(f"Audio length must be greater than unit length ({unit_length}).")

        # Chia file audio thành các đoạn nhỏ
        nums_of_samples = len(audio) // unit_length
        samples_split = [audio[i * unit_length: (i + 1) * unit_length] for i in range(nums_of_samples)]

        # Lưu các đoạn thành các file WAV tạm thời
        temp_wav_paths = []
        for i, sample in enumerate(samples_split):
            temp_path = os.path.join(save_dir, f"temp_sample_{i}.wav")
            sf.write(temp_path, sample, sr)  # Use soundfile to write WAV
            temp_wav_paths.append(temp_path)

        # Dự đoán từng đoạn bằng predict_new30s
        segment_preds, segment_classes = predict_new30s(temp_wav_paths, model_path, save_dir)

        # Lấy kết quả dự đoán bằng phương pháp voting
        pred_index = max(segment_preds, key=segment_preds.count)
        pred_class = class_list[pred_index]

        # Lưu kết quả
        y_pred_index.append(pred_index)
        y_pred_class.append(pred_class)

        # Xóa các file WAV tạm thời
        for temp_path in temp_wav_paths:
            os.remove(temp_path)

    return y_pred_index, y_pred_class



# Tải mô hình đã huấn luyện
try:
    model_path = r'music_genre_classifier.h5'  # Path to your model
except Exception as e:
    st.error("Error loading model. Please check the path.")
    st.stop()



# Giao diện Streamlit
st.title("Music Genre Classification")
st.markdown("Upload a WAV or MP3 file to predict its genre.")

uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Lưu file tạm thời
    temp_audio_path = "temp_audio" + (".wav" if uploaded_file.name.endswith(".wav") else ".mp3")
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    # Nếu là MP3, chuyển đổi sang WAV
    if temp_audio_path.endswith(".mp3"):
        temp_wav_path = temp_audio_path.replace(".mp3", ".wav")
        mp3_2_wav(temp_audio_path, temp_wav_path)
        temp_audio_path = temp_wav_path

    # Dự đoán thể loại
    with st.spinner("Predicting..."):
        try:
            if not os.path.exists("temp_images"):
                os.makedirs("temp_images")
            # Gọi hàm predict_new
            predictions, predicted_classes = predict_new(
                audio_dir=[temp_audio_path],
                model_path=model_path,
                unit_length=661500,  # Default unit length
                save_dir="temp_images"  # Temporary directory for saving images
            )

            predicted_genre = predicted_classes[0]
            st.success(f"The predicted genre of the song is: **{predicted_genre}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Xóa file tạm thời
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
else:
    st.info("Please upload a WAV or MP3 file to predict its genre.")

# D:
# cd downloads\download 20241215
# streamlit run app - DS201.py