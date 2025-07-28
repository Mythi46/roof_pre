# 📖 エキスパート改良版 使用説明書
# Expert Improved Version User Manual

## 🎯 このファイルについて

このフォルダには、エキスパートの診断に基づいて改良された衛星画像セグメンテーション検出システムの日本語版が含まれています。

## 📁 ファイル構成

```
japanese_version/
├── README_JP.md                           # プロジェクト概要（日本語）
├── 使用説明書_JP.md                       # この説明書
├── satellite_detection_expert_final_JP.ipynb  # 完全版ノートブック
└── quick_test_expert_improvements_JP.py   # 高速テストスクリプト
```

## 🚀 使用方法

### 方法1: 高速テスト（推奨）
```python
# Google Colabで以下を実行
!wget https://your-repo/japanese_version/quick_test_expert_improvements_JP.py
exec(open('quick_test_expert_improvements_JP.py').read())
```

### 方法2: 完全版ノートブック
1. `satellite_detection_expert_final_JP.ipynb`をGoogle Colabにアップロード
2. セルを順番に実行
3. 結果を確認

### 方法3: コピー&ペースト
1. `quick_test_expert_improvements_JP.py`の内容をコピー
2. Google Colabの新しいセルにペースト
3. 実行

## 🎯 エキスパート改良のポイント

### ❌ 元の問題 → ✅ 改良解決策

| 問題 | 元バージョン | エキスパート改良版 |
|------|-------------|------------------|
| **クラス重み** | data.yamlに記述（効果なし） | model.train()に直接渡す |
| **重み計算** | [1.4,1.2,1.3,0.6]（手動） | 有効サンプル数法（自動） |
| **解像度** | 訓練640/検証896（不一致） | 統一768（一致） |
| **データ強化** | Mosaic=0.8（有害） | Mosaic=0.25+Copy-Paste=0.5 |
| **学習率** | 線形減衰（単純） | コサイン退火+AdamW（現代的） |
| **推論** | 基本推論 | TTA+タイル推論（高級） |

## 📊 予想される改良効果

- **mAP50**: 3-6ポイント向上
- **クラスバランス**: F1標準偏差の大幅改善
- **訓練安定性**: より滑らかな収束曲線
- **エッジ品質**: より精確なセグメンテーション

## 🔧 設定調整ガイド

### メモリ不足の場合
```python
# 訓練設定で以下を調整
batch=8,            # 16から8に削減
imgsz=640,          # 768から640に削減
```

### 訓練時間を短縮したい場合
```python
epochs=30,          # 60から30に削減
patience=10,        # 20から10に削減
```

### 特定クラスの性能が悪い場合
```python
# 重みを手動調整
cls_weights[class_id] *= 1.5  # 該当クラスの重みを増加
```

## 📈 結果の確認方法

### 1. 訓練結果
```
runs/segment/expert_quick_test_JP/
├── weights/
│   ├── best.pt          # 最良モデル
│   └── last.pt          # 最終モデル
├── results.csv          # 訓練ログ
├── results.png          # 訓練曲線
└── confusion_matrix.png # 混同行列
```

### 2. 性能指標
- **mAP50**: 0.6以上（良好）、0.7以上（優秀）
- **各クラスF1**: 0.5以上（可接受）、0.7以上（優秀）
- **クラスバランス**: F1標準偏差 < 0.15

### 3. 可視化
```python
# 訓練曲線表示
from IPython.display import Image
Image('runs/segment/expert_quick_test_JP/results.png')

# 混同行列表示
Image('runs/segment/expert_quick_test_JP/confusion_matrix.png')
```

## 🔍 トラブルシューティング

### よくある問題と解決策

#### 1. 「CUDA out of memory」エラー
```python
# 解決策: バッチサイズと画像サイズを削減
batch=8,
imgsz=640,
```

#### 2. 訓練が非常に遅い
```python
# 解決策: ワーカー数を削減
workers=2,  # デフォルト8から削減
```

#### 3. 特定クラスの検出率が低い
```python
# 解決策: 該当クラスの重みを増加
cls_weights[class_id] *= 1.5
```

#### 4. データセットダウンロードエラー
```python
# 解決策: APIキーを確認
api_key="YOUR_CORRECT_API_KEY"
```

## 💡 最適化のヒント

### GPU使用時
- `amp=True`（混合精度）を有効にする
- `batch=16`または`batch=32`を使用
- `workers=8`でデータローディングを高速化

### CPU使用時
- `batch=4`に削減
- `workers=2`に削減
- `epochs=20`で短時間テスト

### 高精度が必要な場合
- `epochs=100`に増加
- `imgsz=1024`に増加（メモリ許可時）
- `augment=True`でTTA推論を使用

## 📞 サポート

### 問題が発生した場合
1. **エラーメッセージを確認**
   - CUDA関連: GPU設定を確認
   - メモリ関連: バッチサイズを削減
   - ファイル関連: パスを確認

2. **システム要件を確認**
   - Python 3.8+
   - CUDA対応GPU（推奨）
   - 8GB+ RAM
   - 10GB+ ディスク容量

3. **設定を調整**
   - メモリ不足: batch, imgsz削減
   - 時間短縮: epochs削減
   - 精度向上: 重み調整

### 連絡先
- GitHub Issues: プロジェクトリポジトリ
- 技術サポート: 開発チーム

## 🎉 成功の指標

以下の条件が満たされれば、改良が成功しています：

### 定量的指標
- mAP50が元バージョンより3+ポイント向上
- 各クラスのF1-Scoreが0.5以上
- 訓練曲線が滑らかに収束

### 定性的指標
- セグメンテーションのエッジがより精確
- 少数クラス（Baren-Land, rice-fields）の検出率向上
- 誤検出の減少

### 実用性指標
- 高解像度画像での安定した推論
- 実際の衛星画像での良好な性能
- デプロイ時の一貫した結果

## 📚 参考資料

- **Cui et al., 2019**: "Class-Balanced Loss Based on Effective Number of Samples"
- **YOLOv8 Documentation**: Ultralytics公式ドキュメント
- **セグメンテーション手法**: 最新の研究動向

---

このエキスパート改良版により、より正確で安定した衛星画像セグメンテーション検出が実現されることを期待しています。🛰️✨
