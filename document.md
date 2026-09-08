# document.md — プロジェクト構成・依存関係・実行方法

本ファイルは，本リポジトリを再現・拡張するために必要な，ディレクトリ構成・プログラム間の依存関係・
実行方法・Git管理上の注意事項をまとめたものである．教材としての思想・実験の理論的説明は
`README.md`および各`exNNN_*.md`を参照すること．

## 1. 各ディレクトリ・ファイルの役割

| パス | 役割 |
| :--- | :--- |
| `ex001_random_baseline.ipynb`〜`ex006_comparison_and_play.ipynb` | 各実験の実装(自己完結，Google Colab対応) |
| `ex001_random_baseline.md`〜`ex006_comparison_and_play.md` | 各実験の解説(目的・手法・実験条件・考察・発展課題) |
| `visualize_result.ipynb` | `outputs/`配下の全結果を横断的に走査・可視化する汎用ツール |
| `figures/exNNN/` | 各実験が保存したグラフ画像 |
| `outputs/` | 各実験の学習結果(`Q.npy`, `log.json`, `eval.json`, `config.json`)．Git管理外 |
| `requirements_numpy.txt` | Python仮想環境`.venv_numpy`の依存ライブラリ一覧 |
| `.venv_numpy/` | `uv`で作成したPython仮想環境．Git管理外 |
| `.vscode/settings.json` | VS Codeが`.venv_numpy`を認識するための設定 |
| `.orders/`, `.reports/` | AIエディタへの作業指示と作業報告の記録 |
| `.ai/` | AI開発共通規約(`ai-dev-kit`)・AIエージェント(`ai-agent`)等のサブモジュール |

## 2. プログラム間の依存関係

各`exNNN_*.ipynb`は，他の実験ファイルを一切importしない自己完結なノートブックであり，
`TicTacToeEnv`・`BaseAgent`系のクラス・`ResultLogger`・`set_seed`・`evaluate`・
`play_human_vs_ai`は全ノートブックに同一のコードとして埋め込まれている(意図的な重複，
「1ファイル=1実験・自己完結」の方針による)．ノートブック間の唯一の連携は，
`outputs/`に保存されたデータの読み込みのみである．

```text
ex001 (RandomAgent)          ex002 (PolicyIteration)      ex003 (ValueIteration)
   │ outputs/…/RandomAgent/     │ outputs/…/PolicyIteration/   │ outputs/…/ValueIteration/
   │   eval.json 等              │   Q.npy, eval.json, log.json │   Q.npy, eval.json, log.json
   │                             │                               ▲ log.json を読み込み
   │                             └───────────────────────────────┘ (収束曲線の比較)
   │
   ▼
ex004 (MonteCarloControl)     ex005 (QLearning)
   │ outputs/…/MonteCarloControl/  │ outputs/…/QLearning/
   │   Q.npy, eval.json, log.json  │   Q.npy, eval.json, log.json
   │                               ▲ log.json を読み込み(学習曲線の比較)
   └───────────────────────────────┘
                    │
                    ▼ ex001〜ex005 の Q.npy / eval.json を読み込み
        ex006 (comparison_and_play)
                    │ outputs/ex006_tictactoe_comparison/
                    ▼
        visualize_result.ipynb (outputs/ 全体を横断的に走査)
                    │
                    ▼ outputs/_comparison/
```

各ノートブック内部の主要な関数・クラスの呼び出し関係は次の通りである(`ex002`の例，
`ex004`・`ex005`も同様の構造)．

```text
TicTacToeEnv.transition_model / enumerate_states
        │
        ▼
policy_evaluation_sweep / q_from_value  ──▶ policy_iteration ──▶ build_q_array
                                                   │
                                                   ▼
                                          TabularQAgent(Q_array)
                                                   │
                                                   ▼
                                              evaluate(agent) ──▶ eval.json
                                                   │
                                                   ▼
                                          ResultLogger(収束曲線) ──▶ log.json
```

## 3. 外部モジュールとの依存関係

サードパーティ製ライブラリは，NumPyとMatplotlib，およびJupyter実行環境
(`ipykernel`, `jupyter`, `nbconvert`)とノートブック構築に用いた`nbformat`のみである．
強化学習アルゴリズム自体はNumPyのみでフルスクラッチ実装しており，Gym等の強化学習ライブラリ・
PyTorch等の深層学習フレームワークは使用していない．
`@.ai/ai-dev-kit/src/machine_learning_utils.py`はPyTorchに依存するため使用せず，
`set_seed`・`ResultLogger`は同等のインタフェースを持つ独自実装を各ノートブックに埋め込んでいる．

## 4. Python環境の構築方法

`README.md`の8節を参照．要約すると，`uv venv .venv_numpy --python 3.11`で仮想環境を作成し，
`requirements_numpy.txt`に基づいて依存ライブラリをインストールする．

## 5. プログラムの実行方法

`README.md`の9節を参照．`jupyter nbconvert --to notebook --execute --inplace`により，
各`exNNN_*.ipynb`をノートブック形式のまま実行し，実行結果(出力セル)をノートブックファイル自体に
保存する．`ex006`は`ex001`〜`ex005`の`outputs/`を，`ex003`は`ex002`の`outputs/`(収束曲線の比較用)を，
`ex005`は`ex004`の`outputs/`(学習曲線の比較用)を前提とするため，番号順に実行する必要がある．

## 6. 実験結果の保存場所

`outputs/{実験名}/{アルゴリズム名}/{ハイパーパラメータ}/{seed}/`以下(詳細はREADME.md 6節)．
比較結果は`outputs/ex006_tictactoe_comparison/`および`outputs/_comparison/`に保存する．
グラフ画像は`figures/exNNN/`に保存する．いずれも`.gitignore`によりGit管理対象外である．

## 7. 文書およびレポートの保存場所

- 教材としての解説: `README.md`, 各`exNNN_*.md`
- 作業指示の記録: `.orders/order_{n:03}.md`
- 作業報告(実験条件・結果・考察): `.reports/report_{n:03}.md`

## 8. 必要なAPIキーや設定ファイル

本教材自体はAPIを一切使用しないため，`tokens.json`は不要である．
ただし，`.ai/ai-agent/`(次の実験計画を検討するAIエージェント)を実行する場合は，
プロジェクトルートの`tokens.json`に`gemini`キーでGemini APIキーを設定する必要がある
(詳細は`.ai/ai-agent/README.md`を参照)．`tokens.json`は`.gitignore`により
Git管理対象外としている．

## 9. Git管理上の注意事項

- `.venv_numpy/`, `__pycache__/`, `.ipynb_checkpoints/`, `datasets/`, `outputs/`, `tokens.json`は
  `.gitignore`によりGit管理対象外とする．
- `.vscode/`は`settings.json`のみGit管理対象とし，それ以外は除外する(`.gitignore`の
  `.vscode/*` / `!.vscode/settings.json`による)．
- 各`exNNN_*.ipynb`は実行結果(セル出力・生成した図の埋め込み)を含んだ状態でコミットし，
  実行せずともGitHub上でノートブックの内容と結果を確認できるようにする．
