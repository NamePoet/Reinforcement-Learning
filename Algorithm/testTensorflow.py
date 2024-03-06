import tensorflow as tf

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 列出所有可用的 GPU 设备
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 确保 TensorFlow 使用第一个 GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # 可见设备必须在运行时设置
        print(e)
else:
    print("No GPU available.")

# 创建一个简单的计算图来在 GPU 上运行
with tf.device('/GPU:0'):
    # 创建一个随机张量
    a = tf.random.uniform([1000, 1000])
    # 执行一个矩阵乘法操作
    b = tf.matmul(a, a)

# 打印结果的一部分来验证计算确实发生了
print("Result shape:", b.shape)
print("Result (partial):", b[:5, :5])
