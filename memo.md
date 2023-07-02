# TODO


- しきい値の調整でselfie2animeのマスクが改善されるかの確認
    - 効果あるかもだけどしきい値どうするの？
- ネットワークをUNetベースにして訓練できるようにする
    - 背景をきれいに出力することが狙い
    - また、Maskが少しはもとの構造を保ちやすくなるかも
    - ついでにモダンな構造にする
        - 軽量さをアピールできる
- Transformed Maskの訓練
    - 現状は全くと行ってよいほど訓練ができていない
    - 解決案はいくつかありそう
        - Transformed Maskに対する制約
            - GANでの学習
            - Reconstruction Lossで職人芸
            - MSPCのような空間的な変形に対する制約
        - ネットワーク構造に対する制約
            - ForegroundとBackGroundと別のネットワークで変形させる
        - タスクに応じてもっと適切な特徴抽出器を用いる
            - selfie2animeならface segmentationをさせたほうが性能が向上しそう

- 比較対象の選定・訓練
    - 比較対象は以下のやつにする
        - CycleGAN (2017)
        - MUNIT (2018)
        - Attention GAN v2 (2019)
        - U-GAT-IT (2020)
        - MSPC (2022)
    - 訓練する予定のタスク（優先度順）
        - horse2zebra
            - 
        - selfie2anime
        - apple2orange
        - cityscapes
        - maps
        - front2side
