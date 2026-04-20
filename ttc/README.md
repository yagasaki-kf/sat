# Ultrasonic Motor Controller (TE/UKAEA SAT)

2 台の Raspberry Pi Pico（各 Pico にモータドライバ D0/D1 が接続されている）を、Tera Term の代わりに Web GUIで統合制御するツールです。

## 主な機能
- 4モータを1画面で制御
- Picoごとに `Control Port (TX)` と `Monitor Port (RX)` を設定
- コマンド形式:
  - `D0動作パルス,D1動作パルス,D0最高速度,D1最高速度,D0減速時速度,D1減速時速度,D0減速開始[%],D1減速開始[%]
- `Emergency Stop` で緊急停止コマンド `S,S` を全Picoへ一括送信
- 入力取り消し `,Q` 送信ボタン
- モニタポート受信値からエンコーダ値（各PicoのD0/D1）を常時表示
- 送受信ログ表示

## インストール
```bash
cd ~/FATandSAT/source/TeraTermControl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 起動
```bash
cd ~/FATandSAT/source/TeraTermControl
source .venv/bin/activate
python3 app.py
```

ブラウザで `http://localhost:8086` を開いてください。

## 使い方
1. `Refresh COM Ports` でポート一覧更新
2. 各Picoカードで `Controller Name`, `Control Port (TX)`, `Monitor Port (RX)`, `Baudrate` を設定
3. `Connect` を押し接続、エンコーダの値が取得できていることを確認
4. D0/D1の各パラメータを入力して `Run D0 + D1`
5. 必要に応じて `Emergency STOP (ALL)` または `Cancel Input (,Q)` を使う
6. `Save Settings` で `config.json` に保存

## 注意
- 改行は受信時は `CR`、送信時 `CR+LF` 相当で送っています（`\r\n`）。
- モニタ受信は1行内の整数を抽出し、先頭2個をD0/D1エンコーダ値として表示します。
- 受信フォーマットが異なる場合は `app.py` の `_monitor_loop` を調整してください。
