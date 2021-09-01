# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 08:50:33 2021

@author: salva
"""

import tensorflow as tf
from tensorflow.python.client import device_lib


class Acceleration(object):
    """
    General hardware acceleration class
    """

    @staticmethod
    def get_acceleration() -> None:
        """
        Function to get and configure the hardware acceleration. It will 
        sequentially try to configure TPU, GPU and CPU.
        """
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print("Device:", tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            device = "TPU"

        except:
            strategy = tf.distribute.get_strategy()
            device ="GPU" if "GPU" in [d.device_type for d in 
                                   device_lib.list_local_devices()] else "CPU"

        print(device, "Number of replicas:", strategy.num_replicas_in_sync)
        return strategy, device