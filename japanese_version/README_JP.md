# 🛰️ 衛星画像セグメンテーション検出 - エキスパート改良版
# Satellite Image Segmentation Detection - Expert Improved Version

## 📋 概要

このプロジェクトは、YOLOv8を基盤とした衛星画像セグメンテーション検出システムです。屋根、農地、水田、裸地の4つのクラスを検出します。

エキスパートの診断に基づき、元のバージョンの重要な問題を解決しました。

## 🎯 エキスパート改良のポイント

### ❌ 元バージョンの問題
1. **クラス重みが効果なし** - data.yamlに書いてもYOLOv8が解析しない
2. **重み設定に根拠なし** - 手動設定[1.4,1.2,1.3,0.6]が実際の分布と不一致
3. **解像度の不一致** - 訓練640検証896でmAPが虚高
4. **Mosaicの過度使用** - 0.8はセグメンテーションに有害（エッジ破綻）
5. **学習率戦略が単純** - コサイン退火未使用
6. **損失重みが不適切** - 検出タスクの例をそのまま流用

### ✅ エキスパート改良方案
1. **自動クラス重み計算** - 有効サンプル数法(Cui et al., 2019)
2. **セグメンテーション対応強化** - Mosaic低減+Copy-Paste追加
3. **統一解像度768** - 訓練・検証・推論で一貫
4. **コサイン退火+AdamW** - より安定した収束
5. **TTA+タイル推論** - 高解像度衛星画像対応
6. **専門評価システム** - 混同行列+クラス別分析

## 📊 予想される改良効果

- **mAP50**: 3-6ポイント向上
- **クラスバランス**: F1標準偏差の大幅改善
- **訓練安定性**: より滑らかな収束曲線
- **エッジ品質**: より精確なセグメンテーションマスク

## 📁 ファイル構成

- `satellite_detection_expert_final_JP.ipynb` - 完全版ノートブック（日本語注釈）
- `quick_test_expert_improvements_JP.py` - 高速テストスクリプト（日本語注釈）
- `README_JP.md` - このファイル

## 🚀 使用方法

### 1. 高速テスト
```python
# quick_test_expert_improvements_JP.py を実行
python quick_test_expert_improvements_JP.py
```

### 2. 完全版ノートブック
```python
# Jupyter Notebookで開く
jupyter notebook satellite_detection_expert_final_JP.ipynb
```

### 3. Google Colabで実行
1. ファイルをColabにアップロード
2. セルを順番に実行
3. 結果を確認

## 🔧 主要改良点の詳細

### 1. 自動クラス重み計算
```python
# 有効サンプル数法による自動計算
beta = 0.999
freq = np.array([counter.get(i, 0) for i in range(num_classes)], dtype=float)
eff_num = 1 - np.power(beta, freq)
cls_weights = (1 - beta) / eff_num
cls_weights = cls_weights / cls_weights.mean()
```

### 2. 統一解像度設定
```python
IMG_SIZE = 768  # 訓練・検証・推論で統一
```

### 3. セグメンテーション対応強化
```python
mosaic=0.25,        # 0.8から大幅削減
copy_paste=0.5,     # セグメンテーション専用強化
```

### 4. 現代的学習率戦略
```python
optimizer='AdamW',   # より安定
lr0=2e-4,           # より低い初期学習率
cos_lr=True,        # コサイン退火
```

## 📈 結果の確認方法

1. **訓練曲線**: `runs/segment/expert_final_v5/results.png`
2. **混同行列**: `runs/segment/expert_final_v5/confusion_matrix.png`
3. **各クラス性能**: ノートブック内の詳細分析
4. **推論結果**: 保存された予測画像

## 💡 トラブルシューティング

### メモリ不足の場合
```python
batch=8,            # バッチサイズを削減
imgsz=640,          # 画像サイズを削減
```

### 訓練が遅い場合
```python
epochs=30,          # エポック数を削減
workers=4,          # ワーカー数を削減
```

### 特定クラスの性能が悪い場合
```python
# 手動で重みを調整
cls_weights[class_id] *= 1.5  # 該当クラスの重みを増加
```

## 📞 サポート

問題が発生した場合：
1. エラーメッセージを確認
2. GPU/CPUの使用状況を確認
3. データセットのパスを確認
4. 必要に応じて設定を調整

## 🎯 期待される結果

このエキスパート改良版により：
- より正確なクラス検出
- 安定した訓練プロセス
- 高品質なセグメンテーション
- 高解像度画像への対応

が実現されることを期待しています。
