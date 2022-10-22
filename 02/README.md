# pytorchのテスト

## sample-1.py
最小二乗法を使ったＸＯＲ問題のトレーニングプログラム。

実行方法

```
python3 sample-1.py
```

実行が終わると、カレントディレクトリにsample-1.pthのネットワークファイル
が出力される

## sample-1-1.py
sample-1.pyが出力したネットワークファイルを読み込んで予測するサンプルプログラム。
ネットワークの出力例もある。
sample-1.pyを実行し、sample-1.pthファイルを出力させた後に実行すること。

実行方法

```
python3 sample1-1.py
```

## sample-2.py
CrossEntropyを使ったＸＯＲ問題のトレーニングプログラム。

実行方法

```
python3 sample-2.py
```

## sample-3.py
簡単なファインチューニングを行うサンプルプログラム。
sample-1.pyを実行し、sample-1.pthファイルを出力させた後に実行すること。

実行方法

```
python3 sample-3.py
```

## sample-4.py
ウェイトとバイアスを出力するサンプルプログラム。
実行方法

```
python3 sample-4.py
```


## sample-5.py
モデルの保存方法(state_dict版)

## sample-5-1.py
sample5で学習したファイルを読み込んで実行する場合

## sample-6.py
weightとbiasがどういう変化をするか調べたサンプル
