import whisper

model = whisper.load_model('medium')

audio = whisper.load_audio('data/gpt.oga')
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio).to(model.device)

_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

with open('result.txt', 'w') as f:
    f.write(result.text)
print(result.text)