import argparse
import os
import shutil
import sys
from pathlib import Path

bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
api_dir = bundle / "api"
sys.path.insert(0, str(api_dir))

def initialise_data(target: Path, legacy: Path | None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if legacy is None or not legacy.is_dir():
        return
    for name in ("rag_app.db", ".env", "credentials.json", "token.json"):
        source = legacy / name
        destination = target / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    source_chroma, destination_chroma = legacy / "chroma_db", target / "chroma_db"
    if source_chroma.is_dir() and not destination_chroma.exists():
        shutil.copytree(source_chroma, destination_chroma)
    # Model assets are external to the installer/Git. Preserve independent GGUF
    # copies from the developer installation when upgrading to desktop.
    source_models, destination_models = legacy / 'models', target / 'models'
    if source_models.is_dir():
        destination_models.mkdir(exist_ok=True)
        for source in source_models.iterdir():
            destination = destination_models / source.name
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The packaged API is deliberately loopback-only. Keeping this constraint
    # in the sidecar itself prevents an installer shortcut or manual invocation
    # from accidentally exposing the unauthenticated local API to the LAN.
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--legacy-data-dir")
    args = parser.parse_args()
    if args.data_dir:
        initialise_data(Path(args.data_dir), Path(args.legacy_data_dir) if args.legacy_data_dir else None)
        os.environ["KONTUR_DATA_DIR"] = args.data_dir
    import uvicorn
    from main import app
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
