python src/trigger_jetson_test.py -m \
    test.img_size=416 \
    test.precision_mode=FP32,FP16,INT8 \
    test.protocol=t1,t2