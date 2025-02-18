from flask import Flask, request, jsonify, Response
import subprocess
import os
import threading
import json
import sounddevice as sd
import numpy as np
import requests
import time

app = Flask(__name__)

def stream_audio(text: str, voice: str, speed: float = 1.0):
    """Stream TTS audio using sounddevice"""
    sample_rate = 24000  # Known sample rate
    audio_started = False
    chunk_count = 0
    total_bytes = 0
    first_chunk_time = None
    all_audio_data = bytearray()

    # Initialize sounddevice stream
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
        blocksize=5120,
        latency='high'
    )
    stream.start()

    try:
        # Make streaming request to upstream TTS server
        response = requests.post(
            "http://localhost:8880/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": "pcm",
                "stream": True
            },
            stream=True,
            timeout=1800
        )
        response.raise_for_status()

        # Process streaming response
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                if not audio_started:
                    audio_started = True
                    first_chunk_time = time.time()
                # Convert chunk to numpy array and play
                audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                stream.write(audio_chunk)
                # Keep track of the audio data
                all_audio_data.extend(chunk)
                chunk_count += 1
                total_bytes += len(chunk)

    except Exception as e:
        print(f"Error in audio streaming: {str(e)}")
    finally:
        stream.stop()
        stream.close()

    return bytes(all_audio_data)

@app.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    try:
        data = request.get_json()
        
        # Validate only required fields
        if not all(key in data for key in ['input', 'voice']):
            return jsonify({
                'error': {
                    'message': 'Missing required fields. Required: input, voice',
                    'type': 'invalid_request_error'
                }
            }), 400

        text = data['input']
        voice = data['voice']
        speed = data.get('speed', 1.0)

        # Stream the audio in a separate thread
        audio_thread = threading.Thread(
            target=stream_audio,
            args=(text, voice, speed)
        )
        audio_thread.start()

        # Return success response
        return jsonify({
            'success': True,
            'message': 'Audio streaming started',
            'details': {
                'text': text,
                'voice': voice
            }
        })

    except Exception as e:
        return jsonify({
            'error': {
                'message': f'Error processing request: {str(e)}',
                'type': 'server_error'
            }
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)
