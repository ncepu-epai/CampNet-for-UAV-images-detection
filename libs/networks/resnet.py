# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import tfplot as tfp


def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def fusion_two_layer(C_i, P_j, scope):
    '''
    i = j+1
    :param C_i: shape is [1, h, w, c]
    :param P_j: shape is [1, h/2, w/2, 256]
    :return:
    P_i
    '''
    with tf.variable_scope(scope):
        level_name = scope.split('_')[1]
        h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
        upsample_p = tf.image.resize_bilinear(P_j,
                                              size=[h, w],
                                              name='up_sample_'+level_name)

        reduce_dim_c = slim.conv2d(C_i,
                                   num_outputs=256,
                                   kernel_size=[1, 1], stride=1,
                                   scope='reduce_dim_'+level_name)

        add_f = 0.5*upsample_p + 0.5*reduce_dim_c

        # P_i = slim.conv2d(add_f,
        #                   num_outputs=256, kernel_size=[3, 3], stride=1,
        #                   padding='SAME',
        #                   scope='fusion_'+level_name)
        return add_f


def add_heatmap(feature_maps, name):
    '''

    :param feature_maps:[B, H, W, C]
    :return:
    '''

    def figure_attention(activation):
        fig, ax = tfp.subplots()
        im = ax.imshow(activation, cmap='jet')
        fig.colorbar(im)
        return fig

    heatmap = tf.reduce_sum(feature_maps, axis=-1)
    heatmap = tf.squeeze(heatmap, axis=0)
    tfp.summary.plot(name, figure_attention, [heatmap])


def resnet_base(img_batch, scope_name, is_training=True):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
    # when use fpn . stride list is [1, 2, 2]
    ####                Camp-Net                 #####
    ####  Multi-layer feature fusion structure   #####
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    add_heatmap(C2, name='Layer2/C2_heat')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
    add_heatmap(C3, name='Layer3/C3_heat')
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
    # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
    add_heatmap(C5, name='Layer5/C5_heat')

    feature_dict = {'C2': end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                    'C3': end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                    'C4': end_points_C4['{}/block3/unit_{}/bottleneck_v1'.format(scope_name, middle_num_units - 1)],
                    'C5': end_points_C5['{}/block4/unit_3/bottleneck_v1'.format(scope_name)],
                    # 'C5': end_points_C5['{}/block4'.format(scope_name)],
                    }

    # feature_dict = {'C2': C2,
    #                 'C3': C3,
    #                 'C4': C4,
    #                 'C5': C5}

    pyramid_dict = {}
    with tf.variable_scope('build_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):

            P5 = slim.conv2d(C5,
                             num_outputs=256,
                             kernel_size=[1, 1],
                             stride=1, scope='build_P5')
            if "P6" in cfgs.LEVLES:
                P6 = slim.max_pool2d(P5, kernel_size=[1, 1], stride=2, scope='build_P6')
                pyramid_dict['P6'] = P6

            pyramid_dict['P5'] = P5

            for level in range(4, 1, -1):  # build [P4, P3, P2]

                pyramid_dict['P%d' % level] = fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                               P_j=pyramid_dict["P%d" % (level+1)],
                                                               scope='build_P%d' % level)
            for level in range(4, 1, -1):
                pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                          num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                          stride=1, scope="fuse_P%d" % level)
    for level in range(5, 1, -1):
        add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_heat' % (level, level))

    # return [P2, P3, P4, P5, P6]
    print("we are in Pyramid::-======>>>>")
    print(cfgs.LEVLES)
    print("base_anchor_size are: ", cfgs.BASE_ANCHOR_SIZE_LIST)
    print(20 * "__")
    return [pyramid_dict[level_name] for level_name in cfgs.LEVLES]
    # return pyramid_dict  # return the dict. And get each level by key. But ensure the levels are consitant
    # return list rather than dict, to avoid dict is unordered



def restnet_head(inputs, is_training, scope_name):
    '''
    :param inputs: [minibatch_size, 7, 7, 256]
    :param is_training:
    :param scope_name:
    :return:
    '''

    with tf.variable_scope('build_fc_layers'):
        ####         Camp-Net        #####
        ####  Detector enhancement   #####
        inputs=attention_module(inputs, 256, 'Residual_1',is_train=is_training)

        inputs = slim.flatten(inputs=inputs, scope='flatten_inputs')
        fc1 = slim.fully_connected(inputs, num_outputs=1024, scope='fc1')

        fc2 = slim.fully_connected(fc1, num_outputs=1024, scope='fc2')
        return fc2

def attention_module(x, ci, name, p=1, t=2, r=1,is_train=True):
    """
    4 residual block in total
    Input:
    --- x: Module input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- ci: Input channels
    --- name: Module name
    --- p: The number of pre-processing Residual Units
    --- t: The number of Residual Units in trunk branch
    --- r: The number of Residual Units between adjacent pooling layer in the mask branch
    Output:
    --- outputs: Module output
    """
    with tf.name_scope(name), tf.variable_scope(name):
        # Pre-processing Residual Units
        with tf.name_scope("pre_processing"), tf.variable_scope("pre_processing"):
            pre_pros = x
            for idx in range(p):
                unit_name = "pre_res_{}".format(idx + 1)
                pre_pros = residual_unit(pre_pros, ci, ci, unit_name)
        # Trunk branch
        with tf.name_scope("trunk_branch"), tf.variable_scope("trunk_branch"):
            trunks = pre_pros
            for idx in range(t):
                unit_name = "trunk_res_{}".format(idx + 1)
                trunks = residual_unit(trunks, ci, ci, unit_name)
        # Post-processing Residual Units
        with tf.name_scope("post_processing"), tf.variable_scope("post_processing"):
            for idx in range(p):
                unit_name = "post_res_{}".format(idx + 1)
                trunks = residual_unit(trunks, ci, ci, unit_name)
    return trunks


def residual_unit(x, ci, co, name, stride=1,is_train=True):
    """

    Implementation of Residual Unit
    Input:
    --- x: Unit input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- ci: Input channels
    --- co: Output channels
    --- name: Unit name
    --- stride: Convolution stride
    Output:
    --- outputs: Unit output
    """
    with tf.name_scope(name), tf.variable_scope(name):
        # Batch Normalization
        bn_1 = batch_normal(x, is_train, "bn_1", tf.nn.relu)
        # 1x1 Convolution
        conv_1 = conv(bn_1, 1, 1, co / 4, 1, 1, "conv_1", relu=False)
        # Batch Normalization
        bn_2 = batch_normal(conv_1, is_train, "bn_2", tf.nn.relu)
        # 3x3 Convolution
        conv_2 = conv(bn_2, 3, 3, co / 4, stride, stride, "conv_2", relu=False)
        # Batch Normalization
        bn_3 = batch_normal(conv_2, is_train, "bn_3", tf.nn.relu)
        # 1x1 Convolution
        conv_3 = conv(bn_3, 1, 1, co, 1, 1, "conv_3", relu=False)
        # Skip connection
        if co != ci or stride > 1:
            skip = conv(bn_1, 1, 1, co, stride, stride, "conv_skip", relu=False)
        else:
            skip = x
        outputs = tf.add(conv_3, skip, name="fuse")
    return outputs

def max_pool(x,k_h,k_w,s_h,s_w,name,padding="VALID"):
        """
        Function for max pooling layer
        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- s_h: Stride in height
        --- s_w: Stride in width
        --- name: Layer name
        --- padding: Padding method, SAME or VALID
        Output:
        --- outputs: Output of the max pooling layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.max_pool(x, [1, k_h, k_w, 1], [1, s_h, s_w, 1], padding)
            # Return layer's output
            return outputs

def batch_normal(x, is_train, name, activation_fn=None):
    """
    Function for batch normalization
    Input:
    --- x: input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- is_train: Is training or not
    --- name: Layer name
    --- activation_fn: Activation function
    Output:
    --- outputs: Output of batch normalization
    """
    with tf.name_scope(name), tf.variable_scope(name):
        outputs = tf.contrib.layers.batch_norm(x,
                                               decay=0.999,
                                               scale=True,
                                               activation_fn=activation_fn,
                                               is_training=is_train)
    return outputs

def conv(x,k_h,k_w,c_o,s_h,s_w,name,relu,group=1,
         bias_term=False,padding="SAME",trainable=True):
        """
        Function for convolutional layer
        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- c_o: Amount of kernels
        --- s_h: Stride in height
        --- s_w: Stride in width
        --- name: Layer name
        --- relu: Do relu or not
        --- group: Amount of groups
        --- bias_term: Add bias or not
        --- padding: Padding method, SAME or VALID
        --- trainable: Whether the parameters in this layer are trainable
        Output:
        --- outputs: Output of the convolutional layer
        """
        with tf.name_scope(name), tf.variable_scope(name):
            # Get the input channel
            input_channel=int(x.get_shape()[-1])
            c_i = input_channel / group
            # Create the weights, with shape [k_h, k_w, c_i, c_o]
            weights = make_cpu_variables("weights", [k_h, k_w, c_i, c_o], trainable=trainable)

            # Create a function for convolution calculation
            def conv2d(i, w):
                return tf.nn.conv2d(i, w, [1, s_h, s_w, 1], padding)

            # If we don't need to divide this convolutional layer
            if group == 1:
                outputs = conv2d(x, weights)
            # If we need to divide this convolutional layer
            else:
                # Split the input and weights
                group_inputs = tf.split(x, group, 3, name="split_inputs")
                group_weights = tf.split(weights, group, 3, name="split_weights")
                group_outputs = [conv2d(i, w) for i, w in zip(group_inputs, group_weights)]
                # Concatenate the groups
                outputs = tf.concat(group_outputs, 3)
            if bias_term:
                # Create the biases, with shape [c_o]
                biases = make_cpu_variables("biases", [c_o], trainable=trainable)
                # Add the biases
                outputs = tf.nn.bias_add(outputs, biases)
            if relu:
                # Nonlinear process
                outputs = tf.nn.relu(outputs)
            # Return layer's output
        return outputs

def make_cpu_variables(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def upsample(x, name, size):
        """
        Function for upsample layer
        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        --- size: Upsample size
        Output:
        --- outputs: Output of the upsample layer
        """
        with tf.name_scope(name):
            outputs = tf.image.resize_bilinear(x, size)
            # Return layer's output
            return outputs