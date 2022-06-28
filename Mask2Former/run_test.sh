cd /workspace/Mask2Former
CUDA_LAUNCH_BLOCKING=1 python demo/demo_Trt_test.py --config-file /workspace/Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml --input /workspace/test_imgs_new --opts MODEL.WEIGHTS /workspace/Mask2Former/checkpoint/model_final_94dc52.pkl
