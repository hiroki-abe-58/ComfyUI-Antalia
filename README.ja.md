# Antalia-2 Mini Turkish TTS for ComfyUI（日本語概要）

トルコ語のテキストを音声にするノードです。公式の **Antalia-2 Mini**（[cloud0day3/antalia-mini](https://huggingface.co/cloud0day3/antalia-mini)、
Apache-2.0）を ComfyUI の中で動かします。

- 7.6M パラメータの小さなモデル（重みは 2 種類合わせて 46 MB）で、合成音声の男性ボイス 1 種類、48 kHz 出力です。CPU でも実時間よりずっと速く
  動きます。
- 数字・日付・時刻・金額・略語・URL・英単語を含む生のトルコ語テキストを、パッケージ内の正規化処理で読み方に変換します。
- ノードは公式パッケージ `antalia-mini` を呼び出すだけで、実行中に何もダウンロードしません。

> **非公式**のノードパックです。作者（PatientDesk AI）や Comfy Org とは関係ありません。英語の [README.md](README.md) が正本です。

## ノード

| ノード | 役割 |
| --- | --- |
| **Antalia-2 Mini Loader** | `models/antalia/<名前>/` からモデルを読み込みます（fp32 / fp16、GPU または CPU） |
| **Antalia-2 Mini Speak (Turkish TTS)** | テキスト → `audio`（モノラル）、`spoken_text`（モデルが読む形のテキスト）、`report` |

- `seed`、`speed`（既定 0.95）、`steps`（既定 8）、`cfg`（既定 2.0）、`sample_rate`（48000 / 24000 / 16000 / 8000）、`eq`（出力 EQ）。
  既定値はリリースの設定と同じです。

## 検証の範囲

Windows 11、ComfyUI v0.42.1、CPU と RTX 5090（詳細は [docs/VERIFICATION.md](docs/VERIFICATION.md)）:

- 7 ケースで、別環境の公式 `Antalia(model=<フォルダ>).say(...)` と**サンプル単位で一致**（CPU 7/7、CUDA 7/7）。
- 実際の ComfyUI キューで、保存した FLAC が公式の音声と 16 ビットの半ステップ以内で一致（48 kHz と 8 kHz、fp16、別シード）。
- クリーンインストールと CI（Ubuntu / Windows、ComfyUI v0.38.0 / v0.42.1、実モデル）。

## インストール

1. ComfyUI v0.38.0 以降、Python 3.10 以上。GPU はなくても動きます。
2. `custom_nodes` に clone して `requirements.txt`（`antalia-mini==1.0.1`）をインストールします。
3. モデル（46 MB）をダウンロードします。リビジョン固定、全ファイルの SHA-256 を確認します:

   ```
   python custom_nodes/ComfyUI-Antalia/tools/download_model.py --models-dir models/antalia
   ```

4. ComfyUI を再起動し、`workflows/gui/speak.json`（48 kHz）または `speak_phone.json`（8 kHz）を開きます。

## 注意

- 合成音声です（モデルカードによると VoxCPM2 で生成した音声で学習し、実在の話者の録音は使っていません）。作者の求めどおり、聞き手には合成
  音声であることを伝えてください。
- CPU と GPU では音声がわずかに異なります（7 ケースで SNR 43〜64 dB の差）。同じ環境では毎回同じ音声になります。

## ライセンス

- このノードパックのコード: MIT。
- `antalia-mini`（pip の依存パッケージ）とモデル（`tools/download_model.py` でのみ取得）: Apache-2.0（PatientDesk AI。[NOTICE](NOTICE) 参照）。
