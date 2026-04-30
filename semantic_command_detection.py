import io
from arduino.app_peripherals.microphone import Microphone
from google.cloud import speech
from google.oauth2 import service_account
from arduino.app_bricks.cloud_llm import CloudLLM, CloudModel
from arduino.app_utils import App
import sys
import time
import threading

credentials= service_account.Credentials.from_service_account_file("google_token.json")
with open(
        "python/semantic_prompt.txt", encoding="utf-8", mode="r"
    ) as system_prompt_file:
        classifier_prompt = system_prompt_file.read()
llm = CloudLLM(model=CloudModel.GOOGLE_GEMINI,system_prompt=classifier_prompt)
client = speech.SpeechClient(credentials=credentials)

# Configure the recognition settings
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
    alternative_language_codes=["es-ES"],
    enable_automatic_punctuation=True,
    model="latest_long",
)
streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True,  # Get partial results as they come in
)

def stream_transcribe():
    """Perform real-time streaming speech recognition from the microphone."""
    print("Listening... (press Ctrl+C to stop)\n")
    # Signal that both the main thread and timer thread can see
    stop_event = threading.Event()
    # def timeout_timer(seconds=5):
    #     stop_event.wait(seconds)
    #     stop_event.set()

    with Microphone() as mic:
        audio_generator = mic.stream()

        # Create the stream of requests
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content.tobytes())
            for content in audio_generator
        )

        # Start the streaming recognition
        responses = client.streaming_recognize(
            config=streaming_config,
            requests=requests,
        )
        
        # Process the responses/ start the timer
        # start_time = time.time()
        # timer_started  = False
        for response in responses:
            if not response.results:
            # Keep searching 
                continue
            
            result = response.results[0]
            if not result.alternatives:
                # print(f"time difference{time.time()- start_time}")
                # if time.time() - start_time > 3:

                #     #User hasn't said anything. Break out and turn off Microphone
                #     break
                # else:
                continue
            #Speech Detected, start 5 second timer
            # if not timer_started:
            #     timer_thread = threading.Thread(target=timeout_timer, args=(7,))
            #     timer_thread.daemon = True
            #     timer_thread.start()
            #     timer_started  = True
                

            # if stop_event.is_set():
            #     print("\nTime's up!")
            #     break
            transcript = result.alternatives[0].transcript

            if result.is_final:
                # Final result - print with confidence score
                confidence = result.alternatives[0].confidence
                print(f"Final:   {transcript} [{confidence:.2f}]")
                if not transcript.strip():
                    continue
                response = llm.chat(transcript)
                print(f"AI: {response}")
                
                
            else:
                # Interim result - overwrite the current line
                sys.stdout.write(f"\rInterim: {transcript}")
                sys.stdout.flush()
# Run the streaming transcription
stream_transcribe()
