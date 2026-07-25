import threading
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class GPUMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._data: List[Dict] = []

    def _collect_metrics(self):
        import amdsmi
        amdsmi.amdsmi_init()

        try:
            devices = amdsmi.amdsmi_get_processor_handles()
            if not devices:
                print("Nenhuma GPU AMD encontrada")
                return

            gpu = devices[0]

            while not self._stop_event.is_set():
                timestamp = datetime.now().isoformat()

                try:
                    vram_info = amdsmi.amdsmi_get_gpu_vram_usage(gpu)
                    vram_used_mb = vram_info["vram_used"] / (1024 * 1024)
                    vram_total_mb = vram_info["vram_total"] / (1024 * 1024)

                    gpu_activity = amdsmi.amdsmi_get_gpu_activity(gpu)
                    gpu_percent = gpu_activity["gfx_activity"]

                    self._data.append({
                        "timestamp": timestamp,
                        "vram_used_mb": vram_used_mb,
                        "vram_total_mb": vram_total_mb,
                        "gpu_percent": gpu_percent
                    })
                except Exception as e:
                    print(f"Erro ao coletar métricas: {e}")

                time.sleep(self.interval)
        finally:
            amdsmi.amdsmi_shut_down()

    def start(self):
        self._stop_event.clear()
        self._data = []
        self._thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def save(self, engine_type: str) -> str:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = data_dir / f"telemetry_{engine_type}_{timestamp}.csv"

        if not self._data:
            print("Sem dados de telemetria para salvar")
            return ""

        fieldnames = ["timestamp", "vram_used_mb", "vram_total_mb", "gpu_percent"]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._data)

        return str(filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Monitor Test")
    parser.add_argument("--engine", default="test")
    parser.add_argument("--duration", type=int, default=10, help="Duração em segundos")
    args = parser.parse_args()

    monitor = GPUMonitor(interval=0.5)
    print(f"Iniciando monitoramento por {args.duration} segundos...")
    monitor.start()

    time.sleep(args.duration)

    monitor.stop()
    filename = monitor.save(args.engine)
    print(f"Telemetria salva em: {filename}")
