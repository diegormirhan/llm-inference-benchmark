import threading, time, csv, argparse
from pathlib import Path # manipular caminhos de arquivo
from datetime import datetime # timestamps
from typing import List, Dict # type hints

class GPUMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval # intervalo de coleta
        self._thread: threading.Thread | None = None # Thread que vai rodar em background
        self._stop_event = threading.Event() # sinal para parar a thread
        self._data: List[Dict] = [] # lista onde guarda as métricas

    # Método privado que coleta métricas
    def _collect_metrics(self) -> None:
        import amdsmi
        amdsmi.amdsmi_init() # inicializa a biblioteca AMD

        try:
            devices = amdsmi.amdsmi_get_processor_handles() # lista de GPUs disponíveis
            if not devices:
                print("Nenhuma GPU encontrada")
                return

            gpu = devices[0] # pega a primeira gpu

            # loop até receber stop
            while not self._stop_event.is_set():
                timestamp = datetime.now().isoformat() # marca o horário

                try:
                    vram_info = amdsmi.amdsmi_get_gpu_vram_usage(gpu) # lê VRAM
                    vram_used_mb = vram_info["vram_used"] # converte bytes -> MB
                    vram_total_mb = vram_info["vram_total"] # converte bytes -> MB

                    gpu_activity = amdsmi.amdsmi_get_gpu_activity(gpu) # lê atividade da GPU
                    gpu_percent = gpu_activity["gfx_activity"] # % de uso

                    # Métricas extras: falha isolada não derruba a amostra inteira
                    power_w = gfx_clock_mhz = mem_clock_mhz = None
                    try:
                        power_w = amdsmi.amdsmi_get_power_info(gpu)["socket_power"]
                    except Exception:
                        pass
                    try:
                        gfx_clock_mhz = amdsmi.amdsmi_get_clock_info(gpu, amdsmi.AmdSmiClkType.GFX)["clk"]
                        mem_clock_mhz = amdsmi.amdsmi_get_clock_info(gpu, amdsmi.AmdSmiClkType.MEM)["clk"]
                    except Exception:
                        pass

                    self._data.append({
                        "timestamp": timestamp,
                        "vram_used_mb": vram_used_mb,
                        "vram_total_mb": vram_total_mb,
                        "gpu_percent": gpu_percent,
                        "power_w": power_w,
                        "gfx_clock_mhz": gfx_clock_mhz,
                        "mem_clock_mhz": mem_clock_mhz
                    }) # salva a amostra
                except Exception as e:
                    print(f"Erro ao coletar métricas: {e}")

                time.sleep(self.interval) # espera antes daprimeira coleta
        finally:
            amdsmi.amdsmi_shut_down() # finaliza a biblioteca

    # Inicia o monitoramento
    def start(self) -> None:
        self._stop_event.clear()
        self._data = []
        self._thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self._thread.start()

    # Para o monitoramento
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    # Salva os dados em csv
    def save(self, engine_type: str) -> str:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = data_dir / f"telemetry_{engine_type}_{timestamp}.csv"

        if not self._data:
            print("Sem dados de telemetria para salvar")
            return ""

        fieldnames = ["timestamp", "vram_used_mb", "vram_total_mb", "gpu_percent",
                      "power_w", "gfx_clock_mhz", "mem_clock_mhz"]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._data)

        return str(filename)

# Teste standalone
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Monitor Test")
    parser.add_argument("--engine", default="test")
    parser.add_argument("--duration", type=int, default=10, help="Duração em segundos")
    args = parser.parse_args()

    monitor = GPUMonitor(interval=0.5)
    monitor.start()
    time.sleep(args.duration)
    monitor.stop()
    filename = monitor.save(args.engine)
    print(f"Telemetria salva em: {filename}")