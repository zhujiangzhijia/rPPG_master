# coding: utf-8
import os
import subprocess

def openface(filepath, outdir):
    # 実行フォルダを指定
    place = os.path.join(os.getcwd(),"OpenFace_2.2.0_win_x64")
    # 実行コマンド
    if os.path.isfile(filepath):
        value = 'FeatureExtraction.exe -f "{}" -out_dir "{}" -2Dfp -tracked'.format(filepath, outdir)
    else:
        value = 'FeatureExtraction.exe -fdir "{}" -out_dir "{}" -2Dfp -tracked'.format(filepath, outdir)
    runcmd = subprocess.check_call(value, cwd=place, shell=True)
    print(runcmd)

if __name__ == "__main__":
    filepath = r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\rPPG_master\OpenFace_2.2.0_win_x64\samples\default.wmv"
    outdir = r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\rPPG_master\OpenFace_2.2.0_win_x64\samples\output"
    openface(filepath, outdir)

