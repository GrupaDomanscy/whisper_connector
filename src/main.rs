use std::process::{exit, Stdio};
use rand::Rng;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::codec::{BytesCodec, FramedRead};

#[derive(serde::Deserialize, serde::Serialize)]
struct SimpleOpenAIResponse {
    text: String,
}

fn error_to_string(err: impl std::error::Error) -> String {
    return format!("{err}");
}

async fn send_request(
    language: String, 
    openai_auth_key: String, 
    file_name: String, 
    file: tokio::fs::File
) -> Result<String, String> {
    let client = reqwest::Client::new();

    let stream = FramedRead::new(file, BytesCodec::new());
    let file_body = reqwest::Body::wrap_stream(stream);

    let file_part = reqwest::multipart::Part::stream(file_body)
        .file_name(file_name)
        .mime_str("audio/mpeg")
        .map_err(error_to_string)?;

    let form = reqwest::multipart::Form::new()
        .text("model", "whisper-1")
        .text("language", language)
        .part("file", file_part);

    let response = client.post("https://api.openai.com/v1/audio/transcriptions")
        .bearer_auth(openai_auth_key)
        .multipart(form)
        .send()
        .await
        .expect("Should return response from server");

    let response = response.error_for_status().map_err(error_to_string)?;

    let text = response.text().await.map_err(error_to_string)?;

    let obj: SimpleOpenAIResponse = serde_json::from_str(&text).map_err(error_to_string)?;
    
    return Ok(obj.text);
}

// result:
// .0 - filename
// .1 - absolute path
fn get_audio_sample_absolute_file_path() -> Result<(String, String), String> {
    let temp_dir = std::env::temp_dir();

    let file_seed: String = rand::thread_rng()
        .sample_iter(rand::distributions::Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();

    let file_name = format!("whisper_connector_audio_sample_{file_seed}.mp3");

    let absolute_file_path = temp_dir.join(&file_name);

    match absolute_file_path.to_str() {
        Some(v) => return Ok((file_name, v.to_string())),
        None => return Err("Could not get temporary audio sample file path.".to_string()),
    }
}

#[tokio::main]
async fn main() {
    let openai_auth_key = match std::env::var("OPENAI_AUTH_KEY") {
        Ok(v) => v,
        Err(_) => {
            println!("Required OPENAI_AUTH_KEY environment variable has not been set.");
            exit(1);
        }
    };

    let cmd_args = gd_terminal_utils::get_cmd_args();

    let language = if cmd_args.len() != 1 {
        "pl".to_string()
    } else {
        match cmd_args[0].as_str() {
            "en" => cmd_args[0].clone(),
            _ => {
                println!("Unknown language: {}. Permitted languages: pl, en.", cmd_args[0]);
                exit(1);
            }
        }
    };

    let cancellation_token = tokio_util::sync::CancellationToken::new();
    let ctrlc_cancellation_token = cancellation_token.clone();
    let _ = ctrlc::set_handler(move || ctrlc_cancellation_token.cancel());

    let (file_name, audio_sample_file_path) = match get_audio_sample_absolute_file_path() {
        Ok(v) => v,
        Err(e) => {
            println!("Error occured: {e}");
            exit(1);
        },
    };

    let _ = std::fs::remove_file(&audio_sample_file_path);

    let cmd = tokio::process::Command::new("ffmpeg")
        .args(&[
            // "-list_devices", "true",
            // "-loglevel", "quiet", 
            "-y",
            "-f", "dshow",
            "-i", "audio=Microphone Array (Intel® Smart Sound Technology for Digital Microphones)",
            &audio_sample_file_path,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn();

    if let Err(e) = &cmd {
        eprintln!("could not spawn ffmpeg process that listens to microphone: {}", e);
        cancellation_token.cancel();
        return
    }

    let mut cmd = cmd.unwrap();

    let mut stdin = tokio::io::stdin();

    tokio::select! {
        _ = cancellation_token.cancelled() => {
            if let Err(e) = cmd.kill().await {
                eprintln!("Failed to kill ffmpeg instance: {}", e);
                exit(1);
            }

            return;
        },
        _ = stdin.read_u8() => {},
    };

    let mut cmd_stdin = match cmd.stdin.take() {
        Some(v) => v,
        None => {
            eprintln!("Unknown error occured while taking STDIN from ffmpeg process.");
            exit(1);
        },
    };

    if let Err(e) = cmd_stdin.write(b"q").await {
        eprintln!("Failed to send 'q' key to ffmpeg instance: {}", e);
        exit(1);
    }

    if let Err(e) = cmd_stdin.flush().await {
        eprintln!("Failed to flush stdin of ffmpeg instance: {}", e);
        exit(1);
    }
    
    if let Err(e) = cmd.wait().await {
        eprintln!("Failed to kill ffmpeg instance: {}", e);
        return;
    }

    let file = match tokio::fs::File::open(audio_sample_file_path).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error occured while trying to read recorded audio sample: {e}");
            exit(1);
        },
    };

    let response = match send_request(language, openai_auth_key, file_name, file).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error occured: {e}");
            exit(1);
        }
    };

    println!("{response}");
}

