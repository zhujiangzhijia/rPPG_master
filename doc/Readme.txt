動画から特徴量を出すプログラムの順番は

face_detection_hana
signal_analysis
peak_find
segment_rri

詳細としては，
face_detection_hana:
顔の肌領域ROIを平均化する
signal_analysis:
各領域の信号から脈波を検出する
peak_find:
脈波からピークを検出し，RRIをとりだす
segment_rri:
RRI波形から，各領域におけるパラメータを算出する


の順で処理します

face_detection_hana
で表情のパラメーターも書き出せます。

動画データはNASにあがっていて、サンプルのみ入れておきます。

メモ
FeatureExtraction.exe -f "C:\Users\akito\Desktop\HassyLab\programs\ippg_shibata_inherit\sample\1_toma_0114.avi" -out_dir "C:\Users\akito\Desktop\HassyLab\programs\ippg_shibata_inherit\sample" -of "test"