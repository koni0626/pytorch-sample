# 画像分類プログラムのサンプル

## data_split.py
0000～XXXXのフォルダに保存された画像を訓練用とテスト用に分類するプログラム。
カレントのdata/trainとdata/valに分類する。
分類は7:3になる。

実行方法

```
python3 data_split.py
```


## mean_std.py
画像フォルダから1000枚画像を選び、RGBの平均値と標準偏差を求める。

実行方法

```
python3 mean_std.py
```

## train.py
訓練用プログラム

実行方法

```
python3 train.py
```

## test.py
188行目に指定した画像ファイルを予測する


実行方法

```
python3 test.py
```
