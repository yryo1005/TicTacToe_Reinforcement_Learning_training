# 3目並べで学ぶ強化学習の基礎 — ランダムプレイからQ学習まで

本リポジトリは，3目並べ(Tic-Tac-Toe)を題材として，ランダムプレイ・方策反復法・価値反復法・
モンテカルロ法・Q学習という順序で強化学習の基礎を段階的に学ぶための教材である．
3目並べの環境をNumPyのみでフルスクラッチ実装し，動的計画法(方策反復法・価値反復法)，
モンテカルロ法，Temporal Difference学習(Q学習)の理論的な違いと収束特性を実験的に比較・可視化し，
評価・対戦機能まで実装することを最終到達点とする．

## 1. リポジトリの思想

1. **骨格固定・差分最小**: 全実験で共通の学習・評価ループの構造(`TicTacToeEnv`, `BaseAgent`,
   `evaluate`, `play_human_vs_ai`, `ResultLogger`, `set_seed`等)を用い，実験ごとに変えるのは
   エージェントの価値更新アルゴリズムのみとする．サードパーティ製ライブラリは
   NumPyとMatplotlibのみに限定し，強化学習ライブラリ(Gym等)は使用せずフルスクラッチで実装する．
2. **1ファイル=1実験・自己完結**: 各実験は1つのGoogle Colab対応ノートブック(`exNNN_*.ipynb`)に
   閉じ，他の実験ファイルをimportしない．共通処理はコピー＆微修正とし，DRY(重複の排除)よりも
   「実験間の差分の可視性」を優先する．実験間でのデータの受け渡しは，コードのimportではなく，
   `outputs/`に保存された学習結果(Q値テーブル等)の読み込みのみによって行う(ex003・ex006で使用)．
3. **数式とコードの対応**: 各実験の解説ファイル(`exNNN_*.md`，各`ipynb`内にも同内容を記載)には，
   状態価値関数 $ V(s) $ や行動価値関数 $ Q(s,a) $ の更新式を数式で明記し，コード中のどの変数・
   どの行がその数式のどの項に対応するかを説明する．
4. **問い→予想→実験→結果→考察→発展のサイクル**: 各実験は，実験前に予想する問いを立て，
   結果・考察・発展課題の回答例は折りたたみ(`<details>`)にすることで，学習者が自分で
   予想・考察を書いてから開ける構成にしている．

### 環境のMDP化に関する設計判断

3目並べは2人零和ゲームであるが，方策反復法・価値反復法(動的計画法)は「既知の遷移確率」を
要求する．そこで本教材では，学習エージェントは常に先手(X)とし，後手(O)は合法手から
一様ランダムに1手を選択する固定方策に従う「環境の一部」とみなす．この定式化のもとでは，
遷移確率 $ p(s'\mid s,a) $ がゲーム規則から解析的に計算できる1エージェント視点の
マルコフ決定過程(MDP)となり，動的計画法(ex002・ex003，環境モデル既知)と
モデルフリー手法(ex004・ex005，同じ相手方策からのサンプリングのみで学習)を公平に比較できる．
ex006では，この前提(学習済みエージェントは先手専用である)を踏まえ，「先手が一様ランダムに
打つ場合に最適に打つ後手」を新たに構築し，これを共通の評価基準として全手法を比較する．

## 2. ファイル構成

```text
./
├── README.md                          # 本ファイル
├── document.md                        # 依存関係・実行方法等のドキュメント
├── requirements_numpy.txt             # 依存ライブラリ一覧
├── .venv_numpy/                       # uvで作成したPython仮想環境(Git管理外)
├── .vscode/settings.json              # VS Code用のPython環境設定
├── ex001_random_baseline.ipynb / .md  # 段階1: 環境実装とランダムプレイ
├── ex002_policy_iteration.ipynb / .md # 段階2: 方策反復法(動的計画法1)
├── ex003_value_iteration.ipynb / .md  # 段階3: 価値反復法(動的計画法2)
├── ex004_monte_carlo.ipynb / .md      # 段階4: モンテカルロ制御
├── ex005_q_learning.ipynb / .md       # 段階5: Q学習(TD学習)
├── ex006_comparison_and_play.ipynb / .md  # 段階6: 総合比較・対人対戦
├── visualize_result.ipynb             # outputs/ 配下の全結果を横断的に可視化する汎用ツール
├── figures/                           # 各実験が保存した図(exNNN/以下)
├── outputs/                           # 各実験の学習結果(Git管理外，詳細は5節)
├── .orders/                           # 作業指示の記録
└── .reports/                          # 作業報告(report_001.md等)
```

## 3. 実験系列の一覧

| 実験 | 実験名 | 手法 | 主な更新式 | 前実験との主な差分 |
| :--- | :--- | :--- | :--- | :--- |
| ex001 | tictactoe_baseline | ランダム方策 | なし | 環境のフルスクラッチ実装と評価基盤の構築 |
| ex002 | tictactoe_dp (Policy Iteration) | 方策反復法 | ベルマン期待方程式による方策評価 + 貪欲方策改善 | 既知の環境モデル(固定ランダム方策の相手)を用いた動的計画法を導入 |
| ex003 | tictactoe_dp (Value Iteration) | 価値反復法 | ベルマン最適方程式 | 方策評価と改善を分離せず，価値関数を直接最適化する点のみ変更 |
| ex004 | tictactoe_mc | モンテカルロ制御($ \epsilon $-greedy, 初回訪問) | エピソード収益 $ G_t $ による $ Q(s,a) $ の標本平均更新 | 環境モデルを不要にし，サンプルエピソードのみから学習する点を追加 |
| ex005 | tictactoe_td (Q学習) | TD学習(off-policy) | TD誤差 $ \delta = r + \gamma \max_{a'}Q(s',a') - Q(s,a) $ による更新 | エピソード終了を待たず1ステップごとにブートストラップ更新する点を追加 |
| ex006 | tictactoe_comparison | 全手法の比較 + Human vs AI | なし | 学習曲線・対戦強度の総合比較(最適な後手に対する評価を含む)と対人対戦UIを追加 |

## 4. 共通ハイパーパラメータ

| パラメータ | 値 | 根拠 |
| :--- | :--- | :--- |
| 割引率 $ \gamma $ | 0.95 | 3目並べは有限ステップの終端ゲームであり $ \gamma=1.0 $ でも収束するが，手数による価値の減衰を可視化しやすくするため段階的な値を採用した |
| DP収束判定閾値 $ \theta $ | 1e-4 | 状態数が2,423(Xの手番かつ非終端)と少なく，十分小さい閾値でも現実的な反復回数(5〜15スイープ)で収束する |
| 探索率 $ \epsilon $(MC・Q学習) | 0.1 | 固定相手(ランダム)に対する探索と活用の標準的なバランス |
| 学習率 $ \alpha $(Q学習) | 0.1 | テーブル型TD学習における標準的な初期値 |
| 学習エピソード数(MC・Q学習) | 50,000 | 到達可能な状態行動対数に対し十分な訪問回数を確保するため(ex006で最適な後手に対して敗率0を達成することを確認済み) |
| 評価エピソード数 | 1,000 | 勝率推定の分散を抑えるため |
| シード数 | 3(0, 1, 2) | 学習曲線の平均・標準偏差を`visualize_result.ipynb`で可視化するため |

## 5. 共通クラス・関数の役割

| 名前 | 役割 |
| :--- | :--- |
| `set_seed(seed)` | Python標準randomとNumPyの乱数シードを固定する(torch非依存の独自実装，理由は8節参照) |
| `ResultLogger` | 学習曲線等の指標をメモリ上に記録し，JSONとして保存・読込する軽量ロガー |
| `TicTacToeEnv` | 3目並べの環境クラス．`reset`, `step`, `get_valid_actions`, `render`に加え，動的計画法用に`transition_model`(既知の環境モデル), `enumerate_states`(到達可能状態の列挙)を持つ |
| `BaseAgent` | 全エージェントの基底クラス(`select_action`, `update`) |
| `RandomAgent` | 合法手から一様ランダムに行動するエージェント(ex001のベースライン) |
| `TabularQAgent` | Q値テーブル(NumPy配列，形状`(3**9, 9)`)に基づき貪欲に行動するエージェント．動的計画法・モンテカルロ法・Q学習いずれの最終方策もこの形式で統一する |
| `HumanAgent` | 標準入力から人間が行動を選択するエージェント |
| `evaluate(agent, opponent, n_episodes, seed)` | 任意の2エージェントを対戦させ，勝率・引き分け率・敗率を測定する |
| `play_human_vs_ai(agent)` | 学習済みエージェントと人間が対人対戦するための対話的インタフェース |

## 6. 結果の保存規則

学習結果は以下の規則で`outputs/`配下に保存し，同一条件の結果が既に存在する場合は再学習をスキップする．

```text
outputs/{実験名}/{アルゴリズム名}/{ハイパーパラメータ}/{seed}/
    ├── Q.npy          # 学習後の行動価値テーブル Q(s,a), 形状(3**9, 9)
    ├── log.json        # ResultLoggerによる学習曲線(訓練時の指標)
    ├── eval.json        # evaluate関数による最終評価(勝率・引き分け率・敗率)
    └── config.json      # そのランの全ハイパーパラメータ・学習時間等のメタデータ
```

`log.json`(訓練時の指標)には，一定間隔ごとに現在の貪欲方策を少数エピソード(300)で
簡易評価した値を記録し，`eval.json`(最終評価)には学習終了後の方策を1,000エピソードという
より多くのエピソードで評価した値を記録する．一般的な教師あり学習における訓練・検証データの
役割を，本教材のオンライン強化学習の設定に沿って「学習中の簡易評価」と「学習後の高精度評価」に
対応させたものである．

`ex006_comparison_and_play.ipynb`の比較結果は`outputs/ex006_tictactoe_comparison/`に，
`visualize_result.ipynb`による横断比較結果は`outputs/_comparison/`に保存する．

## 7. 使い方

1. 「8. 環境構築」に従って`.venv_numpy`を構築する．
2. `ex001_random_baseline.ipynb`から順に，対応する`exNNN_*.md`を読みながら実行する．
   各ノートブックの「実験前に考える」の問いに対して自分の予想を書いてから，実験結果のセルを実行し，
   「実験後に考える」「考察」の折りたたみを開いて答え合わせをすることを推奨する．
3. 全ての実験ノートブックの実行後，`visualize_result.ipynb`を実行すると，`outputs/`配下の
   全条件を横断的に比較できる．
4. 学習済みエージェントと対人対戦したい場合は，`ex006_comparison_and_play.ipynb`末尾の
   `RUN_INTERACTIVE`を`True`に変更して実行する(自動実行時に入力待ちで停止しないよう，既定はFalseとしている)．

## 8. 環境構築

Python仮想環境は`uv`で管理し，`.venv_numpy`として作成する．

```bash
uv venv .venv_numpy --python 3.11
uv pip install --python .venv_numpy/bin/python -r requirements_numpy.txt
.venv_numpy/bin/python -m ipykernel install --user --name=venv_numpy --display-name="Python (TicTacToe RL, NumPy)"
```

依存ライブラリは`requirements_numpy.txt`に記載の通り，NumPy・Matplotlib・Jupyter関連
(`ipykernel`, `jupyter`, `nbconvert`, `nbformat`)のみである．強化学習アルゴリズム自体は
NumPyのみでフルスクラッチ実装しており，Gym等の強化学習ライブラリは使用していない．

グラフ中の日本語ラベルを正しく表示するため，各ノートブックは`matplotlib`のフォントとして
`IPAexGothic`を指定している．Google Colab等，同フォントが存在しない環境で実行する場合は，
日本語フォントを別途インストールした上で該当箇所のフォント名を変更すること．

`.ai/ai-dev-kit/src/machine_learning_utils.py`の`set_seed`はPyTorchに依存しているが，
本教材はNumPy/Matplotlibのみで完結させる方針であるため，同等のインタフェースをtorchに
依存しない形で各ノートブック内に独自定義している．同様に，`machine_learning.md`が定める
`load_dataloader`・`load_model`・`iteration`・`epoch`等のPyTorchの教師あり学習を前提とした
関数群は，本教材(テーブル型強化学習)には適用対象が存在しないため使用していない．

## 9. Notebookの実行(動作確認)方法

```bash
.venv_numpy/bin/jupyter nbconvert --to notebook --execute --inplace \\
    --ExecutePreprocessor.kernel_name=venv_numpy ex001_random_baseline.ipynb
```

他の`exNNN_*.ipynb`, `visualize_result.ipynb`についても同様のコマンドで実行できる．
`ex006_comparison_and_play.ipynb`は`ex001`〜`ex005`の`outputs/`が存在することを前提とするため，
番号順に実行すること．
