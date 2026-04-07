# VectorLoRA + distill loss
Hypothesis: Dav2 thuần không cho feature tốt trên ảnh nội soi, tận dụng quá trình train để finetune DaV2 encoder nhưng không muốn nó học nhiều thông tin 2D.
lora_lr = 0.1 * base_lr
## run 1: 
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_5c5b087acac6418d9756e8c1f3a47ecd distill_weight = 0.01
test_stu_dice_end = 0.861, max_dict = 0.865

## run 2:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_5aac4fbbdd4348729d156f4fb7abb006 distill_weight = 0.1
test_stu_dice_end = 0.863, max_dict = 0.871

# relational loss
Hypothesis: Thêm relational loss để student học được mối quan hệ giữa các pixel trong ảnh, giúp cải thiện khả năng phân đoạn.
relational_weight = 0.1
## run 1:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_93e5bc2182e8491db75afc095bd77f70 seed=1111
test_stu_dice_end = 0.863, max_dict = 0.869

## run 2:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_0415458629d541f59154c603031b8b56 seed=3333
test_stu_dice_end = 0.85785, max_dict = 0.870

# Run 600 epoch:
Hypothesis: Vì thấy code DEMT_DaV2Fision_addDepthTrainSignal.py kết thúc ở 300 epoch thì có dice là 0.8647, còn run 350 epoch thì dice là 0.877, nên có thể run thêm epoch sẽ cho kết quả tốt.
## run 1:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_f9fbdcf4cac44b32ba6ca9aa36da8f9c seed=1111
test_stu_dice_end = 0.859, max_dict = 0.867

## run 2:
https://dagshub.com/AkiyaNguyen/polyp_segmentation/experiments#/experiment/m_deed0d2c9b8d489bb054ec0dc3fb7aca seed=1111
test_stu_dice_end = 0.8675, max_dict = 0.876

# scheduler:
Công thức hiện tại theo RD-net là $1-x^{power}$. Cảm giác là dùng công thức $(1 - x) ^ {power}$ không tốt hơn. 