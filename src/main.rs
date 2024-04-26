use std::{io::BufRead, process::{exit, Stdio}};
use rand::Rng;
use tokio::io::{AsyncReadExt, AsyncWriteExt, Stdout};
use tokio_util::codec::{BytesCodec, FramedRead};

#[derive(serde::Deserialize, serde::Serialize)]
struct SimpleOpenAIResponse {
    text: String,
}

fn error_to_string(err: impl std::error::Error) -> String {
    return format!("{err}");
}

async fn get_audio_devices() -> Result<Vec<String>, String> {
    let executed_cmd = tokio::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        .stderr(Stdio::piped())
        .spawn();

    let mut executed_cmd = match executed_cmd {
        Ok(v) => v,
        Err(e) => return Err(format!("{e}")),
    };

    match executed_cmd.wait().await {
        Err(e) => return Err(format!("{e}")),
        _ => {},
    };

    let mut stderr = match executed_cmd.stderr.take() {
        None => return Err(format!("failed to get stderr from process")),
        Some(v) => v,
    };
    
    let mut output_str = String::new();
    match stderr.read_to_string(&mut output_str).await {
        Err(e) => return Err(format!("{e}")),
        _ => {}
    };

    let output_lines = output_str.lines().into_iter().map(|ele| ele.to_string()).collect::<Vec<String>>();

    let mut devices: Vec<String> = Vec::new();

    for output_line in output_lines {
        if !output_line.contains("dshow @") || 
        output_line.contains("]  Alternative name \"") ||
        !output_line.contains(" (audio)")
        { continue; }

        let start_idx = output_line.find(" \"");
        let end_idx = output_line.find("\" ");

        if start_idx.is_none() || end_idx.is_none() {
            continue;
        }

        let start_idx = start_idx.unwrap() + 2;
        let end_idx = end_idx.unwrap();

        if start_idx > end_idx {
            return Err(format!("malformed line returned from ffmpeg, parsing error: \"{}\"", output_line));
        }

        let device_name = &output_line[start_idx..end_idx];

        devices.push(device_name.to_string());
    }

    return Ok(devices);
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

async fn execute_parse_command(openai_auth_key: String, language: String, audio_device: String) -> Result<String, String> {
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
            "-i", format!("audio={audio_device}").as_str(),
            &audio_sample_file_path,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn();

    if let Err(e) = &cmd {
        cancellation_token.cancel();
        return Err(format!("could not spawn ffmpeg process that listens to microphone: {}", e));
    }

    let mut cmd = cmd.unwrap();

    let mut stdin = tokio::io::stdin();

    tokio::select! {
        _ = cancellation_token.cancelled() => {
            if let Err(e) = cmd.kill().await {
                return Err(format!("Failed to kill ffmpeg instance: {}", e));
            }

            return Ok(String::new());
        },
        _ = stdin.read_u8() => {},
    };

    let mut cmd_stdin = match cmd.stdin.take() {
        Some(v) => v,
        None => return Err(format!("Unknown error occured while taking STDIN from ffmpeg process."))
    };

    if let Err(e) = cmd_stdin.write(b"q").await {
        return Err(format!("Failed to send 'q' key to ffmpeg instance: {}", e));
    }

    if let Err(e) = cmd_stdin.flush().await {
        return Err(format!("Failed to flush stdin of ffmpeg instance: {}", e));
    }

    if let Err(e) = cmd.wait().await {
        return Err(format!("Failed to kill ffmpeg instance: {}", e));
    }

    let file = match tokio::fs::File::open(audio_sample_file_path).await {
        Ok(v) => v,
        Err(e) => return Err(format!("Error occured while trying to read recorded audio sample: {e}"))
    };

    let response = match send_request(language, openai_auth_key, file_name, file).await {
        Ok(v) => v,
        Err(e) => return Err(format!("{e}")),
    };

    return Ok(response);
}

#[tokio::main]
async fn main() {
    let cmd_args = gd_terminal_utils::get_cmd_args();

    if cmd_args.len() < 1 {
        eprintln!("Expected at least 1 argument, received {}.", cmd_args.len());
        exit(1);
    }

    match cmd_args[0].as_str() {
        "devices" => {
            let audio_devices = match get_audio_devices().await {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    exit(1);
                }
            };

            let mut i = 1;

            for device in audio_devices {
                println!("{i}. {device}");
                i += 1;
            }
        },
        "transcribe" => {
            let openai_auth_key = match std::env::var("OPENAI_AUTH_KEY") {
                Ok(v) => v,
                Err(_) => {
                    println!("Required OPENAI_AUTH_KEY environment variable has not been set.");
                    exit(1);
                }
            };

            if cmd_args.len() != 3 {
                eprintln!("Expected 3 arguments. Usage: whisper_connector.exe transcribe [language] [audio_device_name]");
                exit(1);
            }

            let language = &cmd_args[1];

            if language != "en" && language != "pl" {
                eprintln!("Unknown language, supported languages: 'en', 'pl'.");
                exit(1);
            }

            let audio_devices = match get_audio_devices().await {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    exit(1);
                }
            };

            let audio_device = cmd_args[2].clone();

            if !audio_devices.contains(&audio_device) {
                eprintln!("This audio device does not exist.");
                exit(1);
            }

            match execute_parse_command(openai_auth_key, language.to_string(), audio_device).await {
                Ok(v) => println!("{}", v),
                Err(e) => eprintln!("{e}"),
            }
            
        },
        _ => {
            println!("Usage:");
            println!("\twhisper_connector.exe transcribe [language] [audio_device_name]");
            println!("\twhisper_connector.exe devices");
        }
    };
}

