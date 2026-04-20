from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import serial
import serial.tools.list_ports
from nicegui import app, ui


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'config.json'


DEFAULT_CONFIG = {
    'picos': [
        {
            'name': 'Pico-A',
            'control_port': '',
            'monitor_port': '',
            'baudrate': 115200,
            'drivers': [
                {'name': 'Stage-X'},
                {'name': 'Stage-Y'},
            ],
        },
        {
            'name': 'Pico-B',
            'control_port': '',
            'monitor_port': '',
            'baudrate': 115200,
            'drivers': [
                {'name': 'Stage-Z'},
                {'name': 'Stage-R'},
            ],
        },
    ]
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding='utf-8')
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding='utf-8')


@dataclass
class DriverCommand:
    pulses: int
    max_speed: int
    decel_speed: int
    decel_start_percent: int


class PicoController:
    def __init__(self, index: int, data: dict, on_encoder_update: Callable[[int, int, int], None], on_log: Callable[[str], None]):
        self.index = index
        self.name = data['name']
        self.control_port = data['control_port']
        self.monitor_port = data['monitor_port']
        self.baudrate = int(data.get('baudrate', 115200))
        self.driver_names = [d.get('name', f'P{index}-D{i}') for i, d in enumerate(data.get('drivers', [{}, {}]))]
        self.on_encoder_update = on_encoder_update
        self.on_log = on_log

        self._control_serial: serial.Serial | None = None
        self._monitor_serial: serial.Serial | None = None
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._lock = threading.Lock()

    def open(self) -> None:
        with self._lock:
            self.close()
            if self.control_port:
                self._control_serial = serial.Serial(self.control_port, self.baudrate, timeout=0.1)
            if self.monitor_port:
                self._monitor_serial = serial.Serial(self.monitor_port, self.baudrate, timeout=0.1)
                self._monitor_stop.clear()
                self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self._monitor_thread.start()
        self.on_log(f'[{self.name}] connected (CTRL={self.control_port}, MON={self.monitor_port}, {self.baudrate}bps)')

    def close(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=0.5)
        self._monitor_thread = None

        if self._monitor_serial:
            try:
                self._monitor_serial.close()
            except Exception:
                pass
            self._monitor_serial = None

        if self._control_serial:
            try:
                self._control_serial.close()
            except Exception:
                pass
            self._control_serial = None

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.is_set():
            try:
                if not self._monitor_serial:
                    break
                raw = self._monitor_serial.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                # Expected monitor format:
                #   [1201923]pos: 3999, 4000
                # where values after "pos:" are D0, D1 absolute encoder positions.
                m = re.search(r'\[\s*-?\d+\s*\]\s*pos:\s*(-?\d+)\s*,\s*(-?\d+)', line, flags=re.IGNORECASE)
                if m:
                    self.on_encoder_update(self.index, 0, int(m.group(1)))
                    self.on_encoder_update(self.index, 1, int(m.group(2)))
                    continue

                # Fallback for legacy/free-form monitor lines:
                nums = [int(n) for n in re.findall(r'-?\d+', line)]
                if len(nums) >= 2:
                    self.on_encoder_update(self.index, 0, nums[0])
                    self.on_encoder_update(self.index, 1, nums[1])
            except Exception as e:
                self.on_log(f'[{self.name}] monitor error: {e}')
                break

    def send_drive_command(self, d0: DriverCommand, d1: DriverCommand) -> str:
        cmd = (
            f'{d0.pulses},{d1.pulses},'
            f'{d0.max_speed},{d1.max_speed},'
            f'{d0.decel_speed},{d1.decel_speed},'
            f'{d0.decel_start_percent},{d1.decel_start_percent}'
        )
        self.send_raw(cmd)
        return cmd

    def send_raw(self, text: str) -> None:
        if not self._control_serial:
            raise RuntimeError(f'{self.name} control port is not connected')
        payload = (text.rstrip('\r\n') + '\r\n').encode('utf-8')
        self._control_serial.write(payload)

    def emergency_stop(self) -> None:
        self.send_raw('S,S')

    def cancel_current_input(self) -> None:
        self.send_raw(',Q')


cfg = load_config()
controllers: list[PicoController] = []
encoder_values = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
encoder_labels: dict[tuple[int, int], ui.label] = {}
connection_status_labels: dict[int, ui.label] = {}
port_meta: dict[str, dict[str, str]] = {}
log_area: ui.log | None = None


def write_log(text: str) -> None:
    if log_area is not None:
        log_area.push(text)


def on_encoder_update(pico_idx: int, driver_idx: int, value: int) -> None:
    encoder_values[(pico_idx, driver_idx)] = value


for i, pico in enumerate(cfg['picos']):
    controllers.append(PicoController(i, pico, on_encoder_update, write_log))


def port_choices() -> dict[str, str]:
    global port_meta
    infos = list(serial.tools.list_ports.comports())
    port_meta = {
        p.device: {
            'description': p.description or '',
            'serial_number': p.serial_number or '',
            'hwid': p.hwid or '',
        }
        for p in infos
    }
    options: dict[str, str] = {}
    for p in infos:
        desc = p.description or '-'
        serial_no = p.serial_number or '-'
        vid = f'{p.vid:04X}' if p.vid is not None else '----'
        pid = f'{p.pid:04X}' if p.pid is not None else '----'
        options[p.device] = f'{p.device} | {desc} | VID:PID {vid}:{pid} | SN {serial_no}'
    return options


def normalize_select_value(options: dict[str, str], value: str | None) -> str | None:
    if value and value in options:
        return value
    return None


def set_port_hint_text(card: dict) -> None:
    ctrl_port = card['ctrl'].value
    mon_port = card['mon'].value
    ctrl_desc = port_meta.get(ctrl_port, {}).get('description', '-') if ctrl_port else '-'
    mon_desc = port_meta.get(mon_port, {}).get('description', '-') if mon_port else '-'
    card['port_hint'].set_text(f'TX: {ctrl_port or "-"} ({ctrl_desc}) | RX: {mon_port or "-"} ({mon_desc})')


def validate_driver_inputs(pico_idx: int, driver_idx: int, driver_ui: dict) -> tuple[DriverCommand | None, list[str]]:
    errors: list[str] = []

    def read_int(key: str, field_name: str) -> int | None:
        value = driver_ui[key].value
        if value is None:
            errors.append(f'{field_name} is empty')
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            errors.append(f'{field_name} must be integer')
            return None

    pulses = read_int('pulses', f'D{driver_idx} Pulses')
    vmax = read_int('vmax', f'D{driver_idx} Max Speed')
    vdec = read_int('vdec', f'D{driver_idx} Decel Speed')
    dstart = read_int('dstart', f'D{driver_idx} Decel Start [%]')

    if pulses is not None:
        if not (-100000 <= pulses <= 100000):
            errors.append('Pulses out of range [-100000, 100000]')
        enc = encoder_values[(pico_idx, driver_idx)]
        delta = pulses - enc
        if not (-20000 <= delta <= 20000):
            errors.append(f'Move too large from encoder ({delta}); limit is +/-20000')
    if vmax is not None and not (5 <= vmax <= 80):
        errors.append('Max Speed out of range [5, 80]')
    if vdec is not None and not (5 <= vdec <= 20):
        errors.append('Decel Speed out of range [5, 20]')
    if dstart is not None and not (10 <= dstart <= 100):
        errors.append('Decel Start [%] out of range [10, 100]')

    if errors or pulses is None or vmax is None or vdec is None or dstart is None:
        return None, errors
    return DriverCommand(pulses=pulses, max_speed=vmax, decel_speed=vdec, decel_start_percent=dstart), []


def refresh_validation_ui(idx: int) -> None:
    if idx >= len(pico_cards):
        return
    card = pico_cards[idx]
    has_error = False
    for d_idx, driver_ui in enumerate(card['drivers']):
        _, errors = validate_driver_inputs(idx, d_idx, driver_ui)
        label = driver_ui['validation']
        if errors:
            has_error = True
            label.set_text('Validation: ' + ' | '.join(errors))
            label.style('color: #b71c1c; font-weight: 700;')
        else:
            pulses = int(driver_ui['pulses'].value)
            enc = encoder_values[(idx, d_idx)]
            delta = pulses - enc
            label.set_text(f'Validation: OK (target={pulses}, encoder={enc}, delta={delta})')
            label.style('color: #1b5e20; font-weight: 700;')
    if has_error:
        card['run_btn'].disable()
    else:
        card['run_btn'].enable()


ui.page_title('Ultrasonic Motor Controller (TE/UKAEA SAT)')
ui.add_head_html('''
<style>
:root {
  --bg: #f6f8fb;
  --card: #ffffff;
  --acc: #0a4f8f;
  --warn: #b71c1c;
}
body { background: linear-gradient(130deg, #f3f7ff 0%, #f4fbf5 100%); }
.panel { background: var(--card); border-radius: 14px; box-shadow: 0 6px 18px rgba(15,40,70,0.08); }
.motor-name { font-weight: 700; color: #213548; }
.encoder { font-size: 1.2rem; font-weight: 700; color: var(--acc); }
</style>
''')


with ui.column().classes('w-full p-3 gap-3'):
    ui.label('Ultrasonic Motor Controller (TE/UKAEA SAT)').classes('text-2xl font-bold')

    with ui.row().classes('w-full items-center gap-2 wrap'):
        ui.button('Emergency STOP (ALL)', color='red', on_click=lambda: [safe_exec(c.emergency_stop, f'{c.name} stop') for c in controllers]).props('unelevated')
        ui.button('Refresh COM Ports', on_click=lambda: rebuild_port_selects())
        ui.button('Save Settings', on_click=lambda: persist_ui_to_config())

    with ui.row().classes('w-full gap-3 no-wrap items-start'):
        pico_cards: list[dict] = []
        for p_idx, pico in enumerate(cfg['picos']):
            ports = port_choices()
            with ui.card().classes('panel p-3 w-1/2 min-w-[420px] max-w-[560px]'):
                ui.label(f'Pico {p_idx + 1}').classes('text-lg')
                name_input = ui.input('Controller Name', value=pico['name']).props('dense outlined')
                ctrl_select = ui.select(
                    options=ports,
                    value=normalize_select_value(ports, pico.get('control_port')),
                    label='Control Port (TX)',
                ).props('dense outlined')
                mon_select = ui.select(
                    options=ports,
                    value=normalize_select_value(ports, pico.get('monitor_port')),
                    label='Monitor Port (RX)',
                ).props('dense outlined')
                baud_input = ui.number('Baudrate', value=pico.get('baudrate', 115200), min=1200, step=1).props('dense outlined').classes('w-32')
                port_hint = ui.label('TX: - | RX: -').classes('text-xs text-slate-600')
                status_label = ui.label('Connection Status: DISCONNECTED').classes('text-sm').style('color: #b71c1c; font-weight: 700;')
                connection_status_labels[p_idx] = status_label

                with ui.row().classes('gap-2'):
                    ui.button('Connect', on_click=lambda idx=p_idx: connect_pico(idx), color='primary')
                    ui.button('Disconnect', on_click=lambda idx=p_idx: disconnect_pico(idx), color='grey')

                driver_rows = []
                for d_idx in range(2):
                    drv = pico['drivers'][d_idx]
                    with ui.card().classes('p-2 bg-slate-50 w-full'):
                        d_name = ui.input(f'D{d_idx} Motor Name', value=drv.get('name', f'D{d_idx}')).props('dense outlined')
                        with ui.row().classes('gap-2 w-full items-end'):
                            pulses = ui.number('Pulses', value=0, min=-100000, max=100000, step=1, format='%.0f').props('dense outlined').classes('w-24')
                            vmax = ui.number('Max Speed [rpm]', value=5, min=5, max=80, step=1, format='%.0f').props('dense outlined').classes('w-24')
                            vdec = ui.number('Decel Speed [rpm]', value=5, min=5, max=20, step=1, format='%.0f').props('dense outlined').classes('w-24')
                            dstart = ui.number('Decel Start [%]', value=80, min=10, max=100, step=1, format='%.0f').props('dense outlined').classes('w-24')
                            enc_label = ui.label('Encoder: 0').classes('encoder')
                            encoder_labels[(p_idx, d_idx)] = enc_label
                        validation_label = ui.label('Validation: waiting input').classes('text-xs text-slate-600')
                    driver_rows.append(
                        {
                            'name': d_name,
                            'pulses': pulses,
                            'vmax': vmax,
                            'vdec': vdec,
                            'dstart': dstart,
                            'validation': validation_label,
                        }
                    )

                with ui.row().classes('gap-2'):
                    run_btn = ui.button('Run D0 + D1', on_click=lambda idx=p_idx: run_dual(idx), color='positive')
                    ui.button('Cancel Input (,Q)', on_click=lambda idx=p_idx: safe_exec(controllers[idx].cancel_current_input, f'{controllers[idx].name} cancel'))

                with ui.expansion('Raw Command (Advanced)', icon='terminal', value=False).classes('w-full'):
                    raw_cmd = ui.input('Raw command', placeholder='ex: 100,100,300,300,100,100,80,80').props('dense outlined')
                    with ui.row().classes('gap-2'):
                        ui.button('Send Raw', on_click=lambda idx=p_idx, box=raw_cmd: send_raw(idx, box.value))
                        ui.button('Send S,S', on_click=lambda idx=p_idx: safe_exec(controllers[idx].emergency_stop, f'{controllers[idx].name} stop'))

                pico_cards.append(
                    {
                        'name': name_input,
                        'ctrl': ctrl_select,
                        'mon': mon_select,
                        'baud': baud_input,
                        'port_hint': port_hint,
                        'run_btn': run_btn,
                        'drivers': driver_rows,
                    }
                )
                card_ref = pico_cards[-1]
                ctrl_select.on('update:model-value', lambda e, card=card_ref: set_port_hint_text(card))
                mon_select.on('update:model-value', lambda e, card=card_ref: set_port_hint_text(card))
                for drv_ui in card_ref['drivers']:
                    drv_ui['pulses'].on('update:model-value', lambda e, idx=p_idx: refresh_validation_ui(idx))
                    drv_ui['vmax'].on('update:model-value', lambda e, idx=p_idx: refresh_validation_ui(idx))
                    drv_ui['vdec'].on('update:model-value', lambda e, idx=p_idx: refresh_validation_ui(idx))
                    drv_ui['dstart'].on('update:model-value', lambda e, idx=p_idx: refresh_validation_ui(idx))
                set_port_hint_text(card_ref)
                refresh_validation_ui(p_idx)

    ui.separator()
    ui.label('Communication Log').classes('text-lg')
    log_area = ui.log(max_lines=300).classes('w-full h-48')



def to_int(v: float | int | None, field: str) -> int:
    if v is None:
        raise ValueError(f'{field} is empty')
    iv = int(v)
    return iv


def ensure_range(value: int, field: str, low: int, high: int) -> None:
    if not (low <= value <= high):
        raise ValueError(f'{field} must be in [{low}, {high}]')


def safe_exec(fn: Callable[[], None], action: str) -> None:
    try:
        fn()
        write_log(f'[OK] {action}')
    except Exception as e:
        write_log(f'[ERR] {action}: {e}')


def update_encoder_labels() -> None:
    for key, label in encoder_labels.items():
        p, d = key
        label.set_text(f'Encoder: {encoder_values[(p, d)]}')
    for idx in range(len(controllers)):
        refresh_validation_ui(idx)


ui.timer(0.2, update_encoder_labels)



def rebuild_port_selects() -> None:
    ports = port_choices()
    for card in pico_cards:
        card['ctrl'].set_options(ports, value=normalize_select_value(ports, card['ctrl'].value))
        card['mon'].set_options(ports, value=normalize_select_value(ports, card['mon'].value))
        set_port_hint_text(card)
    write_log('[INFO] COM port list refreshed')



def persist_ui_to_config() -> None:
    for idx, card in enumerate(pico_cards):
        cfg['picos'][idx]['name'] = card['name'].value or f'Pico-{idx}'
        cfg['picos'][idx]['control_port'] = card['ctrl'].value or ''
        cfg['picos'][idx]['monitor_port'] = card['mon'].value or ''
        cfg['picos'][idx]['baudrate'] = int(card['baud'].value or 115200)
        for d in range(2):
            cfg['picos'][idx]['drivers'][d]['name'] = card['drivers'][d]['name'].value or f'D{d}'
    save_config(cfg)
    write_log('[OK] settings saved to config.json')



def sync_controller_from_ui(idx: int) -> None:
    card = pico_cards[idx]
    ctrl = controllers[idx]
    ctrl.name = card['name'].value or ctrl.name
    ctrl.control_port = card['ctrl'].value or ''
    ctrl.monitor_port = card['mon'].value or ''
    ctrl.baudrate = int(card['baud'].value or 115200)
    ctrl.driver_names = [card['drivers'][0]['name'].value or 'D0', card['drivers'][1]['name'].value or 'D1']



def update_connection_status(idx: int, connected: bool, detail: str = '') -> None:
    label = connection_status_labels.get(idx)
    if label is None:
        return
    text = 'Connection Status: CONNECTED' if connected else 'Connection Status: DISCONNECTED'
    if detail:
        text = f'{text} ({detail})'
    color = '#1b5e20' if connected else '#b71c1c'
    label.set_text(text)
    label.style(f'color: {color}; font-weight: 700;')



def connect_pico(idx: int) -> None:
    sync_controller_from_ui(idx)
    c = controllers[idx]
    if not c.control_port or not c.monitor_port:
        update_connection_status(idx, False, 'select CTRL/RX ports')
        write_log(f'[WARN] {c.name}: select both Control Port (TX) and Monitor Port (RX) before connect')
        return
    if c.control_port == c.monitor_port:
        update_connection_status(idx, False, 'TX and RX must be different ports')
        write_log(f'[WARN] {c.name}: TX and RX must be different COM ports')
        return
    try:
        c.open()
        update_connection_status(idx, True, f'CTRL={c.control_port}, MON={c.monitor_port}')
        write_log(f'[OK] {c.name} connect')
    except Exception as e:
        update_connection_status(idx, False)
        write_log(f'[ERR] {c.name} connect: {e}')



def disconnect_pico(idx: int) -> None:
    c = controllers[idx]
    safe_exec(c.close, f'{c.name} disconnect')
    update_connection_status(idx, False)



def run_dual(idx: int) -> None:
    card = pico_cards[idx]
    c = controllers[idx]
    try:
        d0, d0_errors = validate_driver_inputs(idx, 0, card['drivers'][0])
        d1, d1_errors = validate_driver_inputs(idx, 1, card['drivers'][1])
        if d0_errors or d1_errors or d0 is None or d1 is None:
            refresh_validation_ui(idx)
            raise ValueError('invalid parameters; fix validation errors before run')
        sent = c.send_drive_command(d0, d1)
        write_log(f'[TX] {c.name}: {sent}')
    except Exception as e:
        write_log(f'[ERR] {c.name} run command: {e}')



def send_raw(idx: int, text: str) -> None:
    c = controllers[idx]
    try:
        if not text.strip():
            raise ValueError('empty raw command')
        c.send_raw(text)
        write_log(f'[TX] {c.name}: {text.strip()}')
    except Exception as e:
        write_log(f'[ERR] {c.name} raw command: {e}')


app_shutdown_called = False


def close_all() -> None:
    global app_shutdown_called
    if app_shutdown_called:
        return
    app_shutdown_called = True
    for c in controllers:
        try:
            c.close()
        except Exception:
            pass


async def on_shutdown() -> None:
    close_all()


app.on_shutdown(on_shutdown)


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(title='Ultrasonic Motor Controller (TE/UKAEA SAT)', reload=False, port=8086)
