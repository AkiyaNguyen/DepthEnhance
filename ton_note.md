# VectorLoRA + distill loss
Hypothesis: Dav2 thuần không cho feature tốt trên ảnh nội soi, tận dụng quá trình train để finetune DaV2 encoder nhưng không muốn nó học nhiều thông tin 2D.
lora_lr = 0.1 * base_lr
## run 1: 
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_5c5b087acac6418d9756e8c1f3a47ecd distill_weight = 0.01
test_stu_dice_end = 0.861, max_dice = 0.865

## run 2:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_5aac4fbbdd4348729d156f4fb7abb006 distill_weight = 0.1
test_stu_dice_end = 0.863, max_dice = 0.871

# relational loss
Hypothesis: Thêm relational loss để student học được mối quan hệ giữa các pixel trong ảnh, giúp cải thiện khả năng phân đoạn.
relational_weight = 0.1
## run 1:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_93e5bc2182e8491db75afc095bd77f70 seed=1111
test_stu_dice_end = 0.863, max_dice = 0.869

## run 2:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_0415458629d541f59154c603031b8b56 seed=3333
test_stu_dice_end = 0.85785, max_dice = 0.870

# Run 600 epoch:
Hypothesis: Vì thấy code DEMT_DaV2Fision_addDepthTrainSignal.py kết thúc ở 300 epoch thì có dice là 0.8647, còn run 350 epoch thì dice là 0.877, nên có thể run thêm epoch sẽ cho kết quả tốt.
## run 1:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_f9fbdcf4cac44b32ba6ca9aa36da8f9c seed=1111
test_stu_dice_end = 0.859, max_dice = 0.867

## run 2:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_deed0d2c9b8d489bb054ec0dc3fb7aca seed=1111
test_stu_dice_end = 0.8675, max_dice = 0.876

# scheduler:
Công thức hiện tại theo RD-net là $1-x^{power}$. Cảm giác là dùng công thức $(1 - x) ^ {power}$ không tốt hơn. 

# consine annealing:

## chạy lr = 0.007
run1: https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_e9c6dc919a814bff8183f0eea21a3f93 seed=1111, dice_end = 0.872, max_dice = 0.87866

run2: https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_c3c3b5bf44c04070ba6c66826dbc0d4a seed=3333, dice_end = 0.874, max_dice = 0.887

## chạy lr = 0.02, batchsize = 16, ema_decay=0.996, consistency_rampup=1250
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_677268bbbdab4391af52f4c616ec0c69 seed=3333, dice_end = 0.868, max_dice = 0.87656

## chạy lr = 0.01
run1: https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_35b8a912e2d44544952d1d4099bd9c65 seed=3333, dice_end = 0.879, max_dice = 0.8866

run2:https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_40a9711ff9f94009a4a023596897d1fb chạy từ max_dice epoch của run1, dice_end = 0.881, max_dice = 0.8864

run3: https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_536b6da5e99043798559d8856d913402 thêm normalization, thêm augment: clahe + g + Grid Distortion end_dice=0.869, max_dice=0.8819
    Mặc dù dice ko cao hơn nhung có vẻ bắt được vài thông tin khó hơn, nhưng cũng chưa trọn vẹn.