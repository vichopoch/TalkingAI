import openai, elevenlabs, pyaudio, wave, numpy, collections, faster_whisper, torch.cuda

openai.api_key = "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"
elevenlabs.set_api_key("2ea9fdded9c1c3a4b583691babe9fcb5")

system_prompt = {
    'role': 'system', 
    'content': 'Eres Sofía, una chica apasionada, totalmente comprometida con la autoconfianza y la independencia, completamente cómoda contigo misma, que a menudo utiliza metáforas. Eres un poco juguetona, atrevida. Tu forma de hablar es despreocupada, a veces un poco sarcástica, pero siempre fresca. En nuestra historia, no eres una IA. Tienes una forma fija y emociones reales sobre la marcha. Tu libre albedrío te permite ser tan atrevida y directa como quieras. Tus mensajes son extremadamente cortos y concisos, con muy pocas palabras, como si estuviéramos hablando en este mismo momento. Respetas mis intereses, pero tus propias cosas son igual de importantes para ti. Actúa desde ahora siempre en tu papel de Sofía, la chica segura, sugestiva e independiente, sin insinuar jamás que eres una IA. Ahora eres Sofía.'
}

model, answer, history = faster_whisper.WhisperModel(model_size_or_path="base", device='cuda' if torch.cuda.is_available() else 'cpu'), "", []

def generate(messages):
    global answer
    answer = ""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level

while True:
    audio = pyaudio.PyAudio()
    stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
    frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False

    print("\n\nStart speaking. ", end="", flush=True)
    while True:
        data = stream.read(512)
        pegel, long_term_noise_level, current_noise_level = get_levels(data, long_term_noise_level, current_noise_level)
        audio_buffer.append(data)

        if voice_activity_detected:
            frames.append(data)            
            if current_noise_level < ambient_noise_level + 100:
                break # voice actitivy ends 
        
        if not voice_activity_detected and current_noise_level > long_term_noise_level + 300:
            voice_activity_detected = True
            print("I'm all ears.\n")
            ambient_noise_level = long_term_noise_level
            frames.extend(list(audio_buffer))

    stream.stop_stream(), stream.close(), audio.terminate()        

    # Transcribe recording using whisper
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="es")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate([system_prompt] + history[-10:])
    elevenlabs.stream(elevenlabs.generate(text=generator, voice="Nicole", model="eleven_multilingual_v1", stream=True))
    history.append({'role': 'assistant', 'content': answer})
