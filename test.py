import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "You know what I realized? Nobody's coming. I spent thirty years waiting for the moment everything would finally make sense. Some sign. Some feeling. Some morning where I'd wake up and think — okay. This is it. This is the one. It never came. And the worst part? I'm not even disappointed anymore. Disappointment means you still believe in something. I don't. I really don't. I just show up. Every day. I put on the shoes. I walk out the door. And for what. For what. Not because it matters. Not because I matter. Just because stopping takes more courage than I have. And isn't that just the saddest thing you've ever heard. I'm too much of a coward to quit and too empty to keep going. So I just. Keep. Going."

wav = model.generate(text, audio_prompt_path="C:/Users/w11/Bojack.wav")
ta.save("C:/Users/w11/Desktop/Dialog/üç.wav", wav, model.sr)
print("Tamamlandi!")