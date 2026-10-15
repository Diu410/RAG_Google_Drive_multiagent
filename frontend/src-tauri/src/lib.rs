use serde::Serialize;
use std::{
    fs::{read_to_string, File},
    net::TcpListener,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tauri::{Manager, RunEvent};

#[cfg(windows)]
use std::os::windows::{io::AsRawHandle, process::CommandExt};
#[cfg(windows)]
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE},
    System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    },
};

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(windows)]
struct Job(HANDLE);
#[cfg(windows)]
unsafe impl Send for Job {}
#[cfg(windows)]
impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

#[derive(Clone, Serialize)]
struct Status {
    state: String,
    base_url: String,
    message: Option<String>,
    log: Option<String>,
}

struct Backend {
    child: Option<Child>,
    access_token: Option<String>,
    #[cfg(windows)]
    job: Option<Job>,
    status: Status,
}
type Shared = Arc<Mutex<Backend>>;

fn available_port() -> Result<u16, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
    listener
        .local_addr()
        .map(|a| a.port())
        .map_err(|e| e.to_string())
}

fn healthy(url: &str) -> bool {
    ureq::get(&format!("{url}/api/health"))
        .timeout(Duration::from_millis(500))
        .call()
        .map(|r| r.status() == 200)
        .unwrap_or(false)
}

fn command(app: &tauri::AppHandle, port: u16) -> Result<(Command, std::path::PathBuf), String> {
    let data_dir = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
    if cfg!(debug_assertions) {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = root.join(".venv/Scripts/python.exe");
        if !python.exists() {
            return Err(format!("Не найден Python: {}", python.display()));
        }
        let mut cmd = Command::new(python);
        cmd.current_dir(&root).args([
            "-m",
            "uvicorn",
            "main:app",
            "--app-dir",
            "api",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ]);
        Ok((cmd, data_dir))
    } else {
        let exe = app
            .path()
            .resource_dir()
            .map_err(|e| e.to_string())?
            .join("kontur-backend.exe");
        if !exe.exists() {
            return Err(format!("Backend sidecar не найден: {}", exe.display()));
        }
        let mut cmd = Command::new(exe);
        if std::env::var_os("LLAMA_SERVER_EXECUTABLE").is_none() {
            cmd.env("LLAMA_SERVER_EXECUTABLE", app.path().resource_dir()
                .map_err(|e| e.to_string())?.join("llama/llama-server.exe"));
        }
        cmd.args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--data-dir",
        ])
        .arg(&data_dir);
        if let Ok(resource_dir) = app.path().resource_dir() {
            let marker = resource_dir.join("legacy-data-location.txt");
            if let Ok(path) = read_to_string(marker) {
                let legacy = std::path::PathBuf::from(path.trim_start_matches('\u{feff}').trim());
                if legacy.is_dir() {
                    cmd.arg("--legacy-data-dir").arg(legacy);
                }
            }
        }
        Ok((cmd, data_dir))
    }
}

#[cfg(windows)]
fn contain(child: &Child) -> Result<Job, String> {
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err("Не удалось создать Windows Job Object".into());
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of_val(&info) as u32,
        ) == 0
            || AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE) == 0
        {
            CloseHandle(job);
            return Err("Не удалось привязать backend к lifecycle приложения".into());
        }
        Ok(Job(job))
    }
}

fn launch(app: tauri::AppHandle, shared: Shared) {
    thread::spawn(move || {
        let port = match available_port() {
            Ok(p) => p,
            Err(e) => {
                shared.lock().unwrap().status = Status {
                    state: "error".into(),
                    base_url: String::new(),
                    message: Some(format!("Не удалось выбрать локальный порт: {e}")),
                    log: None,
                };
                return;
            }
        };
        let base = format!("http://127.0.0.1:{port}");
        let (mut cmd, data_dir) = match command(&app, port) {
            Ok(c) => c,
            Err(e) => {
                shared.lock().unwrap().status = Status {
                    state: "error".into(),
                    base_url: base,
                    message: Some(e),
                    log: None,
                };
                return;
            }
        };
        let log_path = data_dir.join("backend.log");
        let log_file = match File::create(&log_path) {
            Ok(f) => f,
            Err(e) => {
                shared.lock().unwrap().status = Status {
                    state: "error".into(),
                    base_url: base,
                    message: Some(format!("Не удалось создать журнал: {e}")),
                    log: None,
                };
                return;
            }
        };
        let stderr = match log_file.try_clone() {
            Ok(f) => f,
            Err(e) => {
                shared.lock().unwrap().status = Status {
                    state: "error".into(),
                    base_url: base,
                    message: Some(e.to_string()),
                    log: None,
                };
                return;
            }
        };
        let access_token = uuid::Uuid::new_v4().to_string();
        cmd.env("KONTUR_API_TOKEN", &access_token);
        cmd.stdout(Stdio::from(log_file))
            .stderr(Stdio::from(stderr))
            .env("KONTUR_DESKTOP", "1");
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                shared.lock().unwrap().status = Status {
                    state: "error".into(),
                    base_url: base,
                    message: Some(format!("Не удалось запустить backend: {e}")),
                    log: None,
                };
                return;
            }
        };
        #[cfg(windows)]
        let job = match contain(&child) {
            Ok(j) => j,
            Err(e) => {
                let mut c = child;
                let _ = c.kill();
                shared.lock().unwrap().status = Status {
                    state: "error".into(),
                    base_url: base,
                    message: Some(e),
                    log: None,
                };
                return;
            }
        };
        {
            let mut backend = shared.lock().unwrap();
            backend.child = Some(child);
            backend.access_token = Some(access_token);
            #[cfg(windows)]
            {
                backend.job = Some(job);
            }
        }
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(180) {
            if healthy(&base) {
                shared.lock().unwrap().status = Status {
                    state: "ready".into(),
                    base_url: base.clone(),
                    message: None,
                    log: None,
                };
                // Readiness is not the end of lifecycle supervision. Keep the
                // owner thread alive so an unexpected backend exit is surfaced
                // in the desktop UI instead of leaving it pointed at a dead API.
                loop {
                    thread::sleep(Duration::from_millis(500));
                    let exited = {
                        let mut guard = shared.lock().unwrap();
                        guard
                            .child
                            .as_mut()
                            .and_then(|c| c.try_wait().ok().flatten())
                    };
                    if let Some(code) = exited {
                        let log = read_to_string(&log_path).ok().map(|s| {
                            s.chars()
                                .rev()
                                .take(8000)
                                .collect::<String>()
                                .chars()
                                .rev()
                                .collect()
                        });
                        shared.lock().unwrap().status = Status {
                            state: "error".into(),
                            base_url: base,
                            message: Some(format!("Backend неожиданно завершился: {code}")),
                            log,
                        };
                        return;
                    }
                }
            }
            let exited = {
                let mut guard = shared.lock().unwrap();
                guard
                    .child
                    .as_mut()
                    .and_then(|c| c.try_wait().ok().flatten())
            };
            if let Some(code) = exited {
                let log = read_to_string(&log_path).ok().map(|s| {
                    s.chars()
                        .rev()
                        .take(8000)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect()
                });
                shared.lock().unwrap().status = Status {
                    state: "error".into(),
                    base_url: base,
                    message: Some(format!("Backend завершился: {code}")),
                    log,
                };
                return;
            }
            thread::sleep(Duration::from_millis(250));
        }
        {
            let mut backend = shared.lock().unwrap();
            if let Some(mut child) = backend.child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            #[cfg(windows)]
            {
                backend.job.take();
            }
        }
        let log = read_to_string(&log_path).ok().map(|s| {
            s.chars()
                .rev()
                .take(8000)
                .collect::<String>()
                .chars()
                .rev()
                .collect()
        });
        shared.lock().unwrap().status = Status {
            state: "error".into(),
            base_url: base,
            message: Some("Backend не ответил за 180 секунд".into()),
            log,
        };
    });
}

#[derive(Serialize)]
struct StatusResponse {
    #[serde(flatten)]
    status: Status,
    access_token: Option<String>,
}

#[tauri::command]
fn backend_status(state: tauri::State<Shared>) -> StatusResponse {
    let backend = state.lock().unwrap();
    StatusResponse {
        status: backend.status.clone(),
        access_token: if backend.status.state == "ready" { backend.access_token.clone() } else { None },
    }
}

#[tauri::command]
fn retry_backend(app: tauri::AppHandle, state: tauri::State<Shared>) {
    let shared = state.inner().clone();
    {
        let mut b = shared.lock().unwrap();
        if b.status.state != "error" {
            return;
        }
        b.access_token = None;
        if let Some(mut child) = b.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        #[cfg(windows)]
        {
            b.job.take();
        }
        b.status = Status {
            state: "starting".into(),
            base_url: String::new(),
            message: None,
            log: None,
        };
    }
    launch(app, shared);
}

pub fn run() {
    let state: Shared = Arc::new(Mutex::new(Backend {
        child: None,
        access_token: None,
        #[cfg(windows)]
        job: None,
        status: Status {
            state: "starting".into(),
            base_url: String::new(),
            message: None,
            log: None,
        },
    }));
    let cleanup = state.clone();
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.show();
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        .manage(state.clone())
        .invoke_handler(tauri::generate_handler![backend_status, retry_backend])
        .setup(move |app| {
            launch(app.handle().clone(), state);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build desktop app")
        .run(move |_, event| {
            if matches!(event, RunEvent::Exit) {
                let mut backend = cleanup.lock().unwrap();
                if let Some(mut child) = backend.child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                #[cfg(windows)]
                {
                    backend.job.take();
                }
            }
        });
}
