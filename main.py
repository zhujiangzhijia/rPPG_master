from src.LandmarkExtract import *

vpath = r"C:\Users\akito\Desktop\HassyLab\programs\rPPG_master\sample\1_toma_0114_Trim.avi"
landmark_data = r"C:\Users\akito\Desktop\HassyLab\programs\rPPG_master\sample\Landmark.csv"

#動画の読み込み
cap = cv2.VideoCapture(vpath)

#Openfaceで取得したLandMark
df = pd.read_csv(landmark_data,header = 0,usecols=range(298,435)).rename(columns=lambda x: x.replace(' ', ''))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(df.shape)

# RGB成分をROIから取り出す
fpath = r"C:\Users\akito\Desktop\testets.xlsx"
ExportRGBComponents(df,cap,fpath)

