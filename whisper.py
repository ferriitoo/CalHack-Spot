import pyaudio
import wave

def record_audio(output_file, duration=10, sample_rate=44100, channels=1):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=1024)
    frames = []

    print("Recording audio...")
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    



record_audio('output.wav', duration=10)


# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

openai.api_key = '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'

audio_file= open("output.wav", "rb")

transcript = openai.Audio.transcribe("whisper-1", audio_file)['text']
# transcript = openai.Audio.transcribe("whisper-1", 'output.wav')

print('TRANSCRIPT: ', transcript)


prompt = f"""

Based on the mapping of characters to each of the keyboard movements: 
mapping = '2':'w', '4': 's', '1': 'a', '3':'d'.
You should build a sequence of movements by concatenating different 
characters based on the number of times each movement is mentioned 
in the <TRANSCRIPT> am gonna give you.

For instance: 1, 1, 3, 3, 3, 2, 2, 4
would be mapped into a result like: 'a a d d d w w s'

This is the <TRANSCRIPT>, which is the input to your response: {transcript} 

Just return the sequence of characters after mapping the transcript 
movements into characters. Do not return any programming code.

"""


response = openai.Completion.create(
            engine="text-davinci-002",  # Use the appropriate engine
            prompt=prompt,
            max_tokens=50,  # You can adjust this to control response length
        )

print('RESPONSE: ', response)

print(response['choices'][0])