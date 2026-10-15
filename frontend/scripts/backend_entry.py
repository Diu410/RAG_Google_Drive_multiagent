import argparse
import os
import sys
from pathlib import Path

bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
api_dir = bundle / "api"
sys.path.insert(0, str(api_dir))

from data_migration import initialise_data

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
