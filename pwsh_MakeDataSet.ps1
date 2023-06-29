#python DataMaker.py -p "ColorRect_Dataset" -r "Img Without MeasureBox and ColorRec - Provided by GN - HNR" -rn "Noise Img - ColorRect" -na "color_rect" -ra 200
python DataMaker.py -p "Bodymark_Dataset" -r "Img Without Bodaymark - Combined HNR results and Provided" -rn "Noise Img - Bodymark" -na "body_mark" -ra 1
python DataMaker.py -p "MeasureBox_Dataset" -r "Img Without MeasureBox and ColorRec - Provided by GN - HNR" -rn "Noise Img - MeasureBox" -na "measure_box" -ra 200
# The Measure box heas fewer samples, so the ration should be much larger
python DataMaker.py -p "Vascular_Dataset_r_plus" -r "Original Img without MeasureBox and Colorec - provided by GN" -rn "Noise Img - Bodymark" -na "vascular_flow" -ra 1000
