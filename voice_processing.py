import os
import uuid
import requests
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment variables for security
STT_API_KEY = os.getenv("STT_API_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")


def speech_to_text(
    audio_path: str, language: str = "uz", has_diarization: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribes an audio file to text using an STT API.

    Args:
        audio_path (str): Path to the audio file (expected to be MP3).
        language (str): Language code for transcription (default: "uz").
        has_diarization (bool): Whether to enable speaker diarization (default: False).

    Returns:
        Tuple[Optional[str], Optional[str]]: (transcript, error_message)
    """
    # Validate audio file format
    if not audio_path.lower().endswith(".mp3"):
        logger.error(f"Invalid file format: {audio_path}. Expected MP3.")
        return None, "Audio file must be in MP3 format"

    # Log audio file details
    file_size = os.path.getsize(audio_path)
    logger.info(f"Sending MP3 file: {audio_path}, size: {file_size} bytes")

    url = "https://back.aisha.group/api/v1/stt/post/"
    headers = {"x-api-key": STT_API_KEY}
    data = {
        "title": f"recording_{uuid.uuid4()}",
        "has_diarization": str(has_diarization).lower(),
        "language": language,
    }
    files = {"audio": open(audio_path, "rb")}

    try:
        response = requests.post(
            url, headers=headers, data=data, files=files, timeout=30
        )
        logger.info(f"STT Response Status: {response.status_code}")
        logger.debug(f"STT Response Content: {response.text}")

        if response.status_code == 200:
            result = response.json()
            transcript = result.get("transcript", "")
            if not transcript:
                logger.warning("Empty transcript received")
                return None, "No speech detected in audio"
            return transcript, None
        else:
            error_msg = response.json().get("error", "Unknown STT error")
            logger.error(
                f"STT Error: Status {response.status_code}, Response: {response.text}"
            )
            return None, error_msg
    except requests.exceptions.Timeout:
        logger.error("STT request timed out")
        return None, "STT request timed out"
    except Exception as e:
        logger.error(f"STT Exception: {str(e)}")
        return None, str(e)
    finally:
        files["audio"].close()


def text_to_speech(
    text: str, language: str = "uz", model: str = "gulnoza", output_dir: str = None
) -> Optional[str]:
    """
    Converts text to an audio file using a TTS API.

    Args:
        text (str): Text to convert to speech.
        language (str): Language code for synthesis (default: "uz").
        model (str): TTS model name (default: "gulnoza").
        output_dir (str): Directory to save the audio file (default: Flask app.config['UPLOAD_FOLDER']).

    Returns:
        Optional[str]: Path to the generated audio file, or None if failed.
    """
    url = "https://back.aisha.group/api/v1/tts/post/"
    headers = {
        "x-api-key": TTS_API_KEY,
        "X-Channels": "stereo",
        "X-Quality": "64k",
        "X-Rate": "16000",
        "X-Format": "mp3",
    }
    data = {"transcript": text, "language": language, "model": model}

    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        logger.info(f"TTS Response Status: {response.status_code}")
        logger.debug(f"TTS Response Content: {response.text}")

        if response.status_code in (200, 201):
            response_data = response.json()
            audio_url = response_data.get("audio_path")
            if not audio_url:
                logger.error("No audio_path in response")
                return None

            audio_response = requests.get(audio_url, timeout=30)
            if audio_response.status_code == 200:
                output_dir = output_dir or os.path.join(os.getcwd(), "audio")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"output_{uuid.uuid4()}.mp3")
                with open(output_file, "wb") as f:
                    f.write(audio_response.content)
                logger.info(f"TTS Output saved: {output_file}")
                return output_file
            else:
                logger.error(
                    f"TTS Audio Fetch Error: Status {audio_response.status_code}"
                )
                return None
        else:
            logger.error(
                f"TTS Error: Status {response.status_code}, Response: {response.text}"
            )
            return None
    except requests.exceptions.Timeout:
        logger.error("TTS request timed out")
        return None
    except Exception as e:
        logger.error(f"TTS Exception: {str(e)}")
        return None
