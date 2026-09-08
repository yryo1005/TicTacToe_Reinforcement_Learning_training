# document.md — プロジェクト構成・依存関係

本ファイルは，本リポジトリのディレクトリ構成とプログラム間の依存関係をまとめたものである．
教材としての思想・理論的説明・使い方は`README.md`および各`exNNN_*.md`を参照すること．

## 1. 各ディレクトリ・ファイルの役割

| パス | 役割 |
| :--- | :--- |
| `ex001_random_baseline.ipynb`〜`ex006_comparison_and_play.ipynb` | 各実験の実装(自己完結，Google Colab対応) |
| `ex001_random_baseline.md`〜`ex006_comparison_and_play.md` | 各実験の解説(目的・手法・実験条件・結果・考察・発展課題) |
| `visualize_result.ipynb` | `outputs/`配下の全結果を横断的に走査・可視化する汎用ツール |
| `figures/exNNN/` | 各実験が保存したグラフ画像 |
| `outputs/` | 各実験の学習結果(`Q.npy`, `log.json`, `eval.json`, `config.json`)．Git管理外 |

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

## 3. 実験結果の保存場所

`outputs/{実験名}/{アルゴリズム名}/{ハイパーパラメータ}/{seed}/`以下(詳細はREADME.md 6節)．
比較結果は`outputs/ex006_tictactoe_comparison/`および`outputs/_comparison/`に保存する．
グラフ画像は`figures/exNNN/`に保存する．いずれも`.gitignore`によりGit管理対象外である．

## 4. Git管理上の注意事項

- `outputs/`, `tokens.json`は`.gitignore`によりGit管理対象外とする．
- 各`exNNN_*.ipynb`は実行結果(セル出力・生成した図の埋め込み)を含んだ状態でコミットし，
  実行せずともGitHub上でノートブックの内容と結果を確認できるようにする．
