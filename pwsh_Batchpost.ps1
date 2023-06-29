$job = Start-Job -Name "Job1" -ScriptBlock {
    python postProces.py -np "Vascular_Dataset_quality" -ep "Vascular_Dataset_Export_N2C_q" -c "G:\Pytorch-QA-MG\checkpoints\Vascular_SmoothL1_N2C\2023-01-13_checkpoint_Vascular_Dataset_epoch_10.pth"
}
Start-Sleep -s 1800
Stop-Job $job
$job = Start-Job -Name "Job1" -ScriptBlock {
    python postProces.py -np "Vascular_Dataset_quality" -ep "Vascular_Dataset_Export_N2N_q" -c "G:\Pytorch-QA-MG\checkpoints\Vascular_SmoothL1_N2N\2023-01-11_checkpoint_Vascular_Dataset_epoch_10.pth"
}
Start-Sleep -s 1800
Stop-Job $job
$job = Start-Job -Name "Job1" -ScriptBlock {
    python postProces.py -np "Measure_Anchor_quality" -ep "Measure_Anchor_EXPORT_N2C_q" -c "G:\Pytorch-QA-MG\checkpoints\Measure_Anchor_SmoothL1_N2C\2023-01-01_checkpoint_Measure_Anchor_Dataset_epoch_10.pth"
}
Start-Sleep -s 1800
Stop-Job $job
$job = Start-Job -Name "Job1" -ScriptBlock {
    python postProces.py -np "Measure_Anchor_quality" -ep "Measure_Anchor_EXPORT_N2N_q" -c "G:\Pytorch-QA-MG\checkpoints\Measure_Anchor_SmoothL1_N2N\2022-12-31_checkpoint_Measure_Anchor_Dataset_epoch_10.pth"
}
Start-Sleep -s 1800
Stop-Job $job
$job = Start-Job -Name "Job1" -ScriptBlock {
    python postProces.py -np "Bodymark_Dataset_quality" -ep "Bodymark_Dataset_export_N2N_no_norm_q" -c "G:\Pytorch-QA-MG\checkpoints\BodyMark\N2N_Costume_Unet_N2N_no_Norm_Smooth\2022-12-17_checkpoint_Bodymark_Dataset_epoch_10.pth"
}
Start-Sleep -s 1800
Stop-Job $job
$job = Start-Job -Name "Job1" -ScriptBlock {
    python postProces.py -np "Bodymark_Dataset_quality" -ep "Bodymark_Dataset_export_N2C_no_norm_q" -c "G:\Pytorch-QA-MG\checkpoints\BodyMark\N2N_Costume_Unet_N2C_no_Norm\2022-12-13_checkpoint_Bodymark_Dataset_epoch_10.pth"
}
Start-Sleep -s 1800
Stop-Job $job